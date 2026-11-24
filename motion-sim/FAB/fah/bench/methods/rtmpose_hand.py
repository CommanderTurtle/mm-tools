"""Adapter: RTMPose hand (https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) -> benchmark prediction format.

Official top-down hand pipeline as released in mmpose's RTMPose ONNX SDK models (the ones rtmlib's `Hand` solution
uses): RTMDet-nano hand detector (320x320, single "hand" class) + RTMPose-m hand5 (256x256 SimCC, 21 keypoints in
COCO-WholeBody-hand order == OpenPose hand order), run with onnxruntime through rtmlib's pre/post-processing.
Two deviations from rtmlib's `Hand`, both to match the SDK pipeline.json / mmpose demo: the pose model gets RGB
(rtmlib feeds BGR, but the model was trained with bgr_to_rgb), and detections are filtered as in mmpose's
topdown demo (score > det_conf 0.3, NMS IoU 0.3).

wholebody=True: RTMW-l (rtmw-dw-x-l 384x288, cocktail14) on YOLOX-m HumanArt person boxes, as rtmlib's
`Wholebody(mode="performance")`; hands = COCO-WholeBody keypoints 91-111 (left) / 112-132 (right), kept if their
mean keypoint score >= hand_conf; bbox = keypoint box padded by bbox_pad * its longer side; handedness from the body.

2D only: joints3d is None (HOT3D pose metrics N/A; detection recall still measured). The hand model predicts no
handedness: is_right is an egocentric image-side prior (box centre right of the image centre), so
handedness_acc for this method measures that prior, not the model. score = detector score.
"""
import os

import cv2
import numpy as np

SDK = "https://download.openmmlab.com/mmpose/v1/projects/"
DET = SDK + "rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
POSE = SDK + "rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
PERSON_DET = SDK + "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
WHOLEBODY = SDK + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"


def find_onnx(root, url):
    for dp, _, fs in os.walk(os.path.join(root, os.path.basename(url)[:-4])):
        if "end2end.onnx" in fs:
            return os.path.join(dp, "end2end.onnx")
    return url  # rtmlib downloads + caches it


class RTMPoseHandMethod:
    name = "rtmpose-m"

    def __init__(self, model_dir=None, det_size=320, pose_size=256, det_conf=0.3, det_nms=0.3, wholebody=False,
                 hand_conf=0.3, bbox_pad=0.05, pose_rgb=True, device="cpu", threads=0, **_):
        import onnxruntime as ort
        from rtmlib import YOLOX, RTMDet, RTMPose

        root = model_dir or os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "rtmpose")  # prepare_mediapipe_rtm.sh
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        if wholebody:
            self.det = YOLOX(find_onnx(root, PERSON_DET), model_input_size=(640, 640), backend="onnxruntime")
            self.pose = RTMPose(find_onnx(root, WHOLEBODY), model_input_size=(288, 384), backend="onnxruntime")
        else:
            self.det = RTMDet(find_onnx(root, DET), model_input_size=(det_size, det_size), backend="onnxruntime")
            self.pose = RTMPose(find_onnx(root, POSE), model_input_size=(pose_size, pose_size), backend="onnxruntime")
        for m in (self.det, self.pose):  # same models, our session options / providers
            m.session = ort.InferenceSession(m.onnx_model, sess_options=so, providers=prov)
        self.det_conf, self.det_nms = det_conf, det_nms
        self.wholebody, self.hand_conf, self.bbox_pad = wholebody, hand_conf, bbox_pad
        self.pose_rgb = pose_rgb  # False: BGR like rtmlib
        self.name = "rtmw-l" if wholebody else "rtmpose-m"

    def detect(self, image_bgr):
        from rtmlib.tools.object_detection.post_processings import nms

        img, ratio = self.det.preprocess(image_bgr)
        dets = self.det.inference(img)[0][0]  # SDK export with NMS baked in: (N, 5) x1 y1 x2 y2 score
        boxes, scores = dets[:, :4] / ratio, dets[:, 4]
        keep = scores > self.det_conf
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes):
            keep = nms(boxes, scores, self.det_nms)
            boxes, scores = boxes[keep], scores[keep]
        return boxes, scores

    def _pose_input(self, image_bgr):
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if self.pose_rgb else image_bgr

    def _wholebody(self, image_bgr):
        persons = self.det(image_bgr)  # rtmlib's YOLOX post-processing (score 0.7, NMS 0.45), as its Wholebody
        if len(persons) == 0:
            return []
        kps, sc = self.pose(self._pose_input(image_bgr), bboxes=persons)
        out = []
        for k, s in zip(kps, sc):
            for is_right, sl in ((False, slice(91, 112)), (True, slice(112, 133))):
                if s[sl].mean() < self.hand_conf:
                    continue
                lo, hi = k[sl].min(0), k[sl].max(0)
                pad = self.bbox_pad * (hi - lo).max()
                out.append(dict(is_right=is_right, score=float(s[sl].mean()), bbox=np.array([*(lo - pad), *(hi + pad)], np.float32),
                                joints2d=k[sl].astype(np.float32), kp_scores=s[sl].astype(np.float32), joints3d=None))
        return out

    def __call__(self, image_bgr, K):
        if self.wholebody:
            return self._wholebody(image_bgr)
        boxes, scores = self.detect(image_bgr)
        if len(boxes) == 0:
            return []
        kps, kp_scores = self.pose(self._pose_input(image_bgr), bboxes=boxes)
        w = image_bgr.shape[1]
        return [dict(is_right=bool((b[0] + b[2]) / 2 > w / 2), score=float(s), bbox=b.astype(np.float32),
                     joints2d=k.astype(np.float32), kp_scores=ks.astype(np.float32), joints3d=None)
                for b, s, k, ks in zip(boxes, scores, kps, kp_scores)]
