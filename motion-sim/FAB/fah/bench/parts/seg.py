"""Part-head quality on held-out labelled crops (scripts/part_seg_eval.py).

Two passes per crop, both with the model's part head (parts_roi; parts_in_mask=False by default: the raw head):
  box mode   the crop's own label box as the only detection (right hand): pixel accuracy / per-class IoU / mIoU of the
             part map against the reference labels (17 classes incl. background; label 255 ignored), and the union of
             the parts against the labelled hand (IoU, boundary F at 2 px) and against the pool's SAM hand mask;
  detector   the full pipeline on the same crop (only where a SAM mask exists): the detection whose instance mask
             overlaps the SAM mask most; its mask and its part union against the SAM mask, and its part map against the
             labels ("detbox_*", detected hands only; "e2e_*": a missed hand counts as background everywhere).
The paper's part-head cells (supplementary part-segmentation table, released heads) are detbox_miou_parts (16 parts, detected hands), n_detected,
detbox_union_iou_sam / detbox_union_bf_sam.

Predictions are kept per crop (predict_crop) so that the metrics can be recomputed from them (score_pool)."""
import cv2
import numpy as np

from bench.seg.evaluate import boundary_f
from bench.seg.rle import decode

N_CLASSES = 17  # background, palm, 3 segments x 5 fingers (train.part_masks)


class BoxDetector:
    """Stands in for the pipeline's detector: the crop's label box as the only hand (right)."""

    def __init__(self, inner):
        self.inner, self.box = inner, None

    def __call__(self, img):
        return self.box[None].astype(np.float32), np.ones(1, np.float32), np.ones(1, bool)


CROP_K = np.array([[640, 0, 320], [0, 640, 320], [0, 0, 1]], np.float32)  # stored crops are 640 px


def _roi(entry):
    return None if entry is None else (int(entry[0]), int(entry[1]), np.asarray(entry[2], np.uint8))


def predict_crop(method, box_det, detector, img, box, with_detector):
    """One crop's stored prediction: {"box": parts_roi of the box-mode hand or None, "dets": [{"mask_det", "parts_roi"}]
    of the detector pass (None if it did not run)}. method: a bench.methods.fah.FAHMethod; box_det: its BoxDetector;
    detector: the pipeline's own detector."""
    box_det.box = box
    method.detector = method.est.detector = box_det  # verify off: the estimator calls its own detector reference
    hs = method(img, CROP_K.copy())
    out = dict(box=_roi(hs[0].get("parts_roi")) if hs else None, dets=None)
    if with_detector:
        method.detector = method.est.detector = detector
        hd = method(img.copy(), CROP_K.copy())  # a new frame for the method's per-frame cache
        out["dets"] = [dict(mask_det=h.get("mask_det"), parts_roi=_roi(h.get("parts_roi"))) for h in hd]
    return out


def _paste(roi, shape, binary=False):
    out = np.zeros(shape, bool if binary else np.uint8)
    if roi is not None:
        x0, y0, r = roi
        h, w = min(r.shape[0], shape[0] - y0), min(r.shape[1], shape[1] - x0)
        out[y0:y0 + h, x0:x0 + w] = (r[:h, :w] > 0) if binary else r[:h, :w]
    return out


def new_acc():
    return dict(conf=np.zeros((N_CLASSES, N_CLASSES)), conf_det=np.zeros((N_CLASSES, N_CLASSES)), conf_e2e=np.zeros((N_CLASSES, N_CLASSES)),
                lists={k: [] for k in ("iou_lab", "bf_lab", "iou_sam", "bf_sam", "det_iou_sam", "det_bf_sam", "detbox_union_iou_sam",
                                       "detbox_union_bf_sam", "det_found", "e2e_union_iou_sam")})


def score_crop(acc, lab, sam, pred):
    """Adds one crop: lab (H, W) uint8 part label (255 ignore), sam (H, W) uint8 SAM hand mask or None, pred = predict_crop's."""
    a, conf = acc["lists"], acc["conf"]
    p = _paste(pred["box"], lab.shape)
    ok = lab != 255
    np.add.at(conf, (lab[ok].astype(int), p[ok].astype(int)), 1)
    un, gl = p > 0, (lab > 0) & ok
    a["iou_lab"].append((un & gl & ok).sum() / max(((un | gl) & ok).sum(), 1))
    a["bf_lab"].append(boundary_f(un & ok, gl))
    if sam is None:
        return
    sm = sam > 127
    a["iou_sam"].append((un & sm).sum() / max((un | sm).sum(), 1))
    a["bf_sam"].append(boundary_f(un, sm))
    best = None
    for d in pred["dets"] or []:
        if d["mask_det"] is not None:
            dm = decode(d["mask_det"]).astype(bool)
            iou = (dm & sm).sum() / max((dm | sm).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, dm, d)
    a["det_found"].append(float(best is not None))  # the detector-pass metrics below are conditional on a detection
    if best is None:  # end to end: a missed hand scores as background everywhere
        np.add.at(acc["conf_e2e"], (lab[ok].astype(int), np.zeros(int(ok.sum()), int)), 1)
        a["e2e_union_iou_sam"].append(0.0)
        return
    a["det_iou_sam"].append(best[0])
    a["det_bf_sam"].append(boundary_f(best[1], sm))
    pu = _paste(best[2]["parts_roi"], sm.shape, binary=True)  # the part union of that detected hand
    a["detbox_union_iou_sam"].append((pu & sm).sum() / max((pu | sm).sum(), 1))
    a["detbox_union_bf_sam"].append(boundary_f(pu, sm))
    pd = _paste(best[2]["parts_roi"], lab.shape)  # its part map vs the labels
    np.add.at(acc["conf_det"], (lab[ok].astype(int), pd[ok].astype(int)), 1)
    np.add.at(acc["conf_e2e"], (lab[ok].astype(int), pd[ok].astype(int)), 1)
    a["e2e_union_iou_sam"].append(a["detbox_union_iou_sam"][-1])


def _iou(c):
    return np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)


def report(acc, parts_in_mask=False):
    """Pool metrics from the accumulated crops."""
    a, conf, conf_det, conf_e2e = acc["lists"], acc["conf"], acc["conf_det"], acc["conf_e2e"]
    iou, iou_d, iou_e = _iou(conf), _iou(conf_det), _iou(conf_e2e)
    return dict(parts_in_mask=bool(parts_in_mask), n=len(a["iou_lab"]),
                pixel_acc=float(np.trace(conf) / max(conf.sum(), 1)), miou=float(iou.mean()),
                miou_parts=float(iou[1:].mean()), iou_per_class=[round(float(x), 3) for x in iou],
                detbox_pixel_acc=float(np.trace(conf_det) / max(conf_det.sum(), 1)), detbox_miou=float(iou_d.mean()),
                detbox_miou_parts=float(iou_d[1:].mean()), detbox_iou_per_class=[round(float(x), 3) for x in iou_d],
                n_mask_evaluable=len(a["det_found"]), n_detected=int(sum(a["det_found"])),
                e2e_miou_parts=float(iou_e[1:].mean()), e2e_pixel_acc=float(np.trace(conf_e2e) / max(conf_e2e.sum(), 1)),
                **{k: (float(np.mean(v)) if v else None) for k, v in a.items()})


def read_label(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
