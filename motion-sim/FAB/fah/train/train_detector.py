"""Fine-tune a hand detector from an explicit checkpoint (2 classes left/right; YOLO pose or segmentation)
on train.build_det_dataset output. Horizontal flips are disabled: they would swap left/right semantics.

    python -m train.train_detector --data $DATA_ROOT/det/hands.yaml --init $FAH_WEIGHTS/shared/detector.pt --out $CKPT_ROOT/detector --name detector \
        --batch "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --device "${TRAIN_DEVICES:?set TRAIN_DEVICES}"
"""
import argparse
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main(args):
    from ultralytics import YOLO

    init = args.init
    model = YOLO(init)
    if args.cfg:  # another architecture (e.g. yolov8s-seg.yaml: hand instance masks), layers matching --init transferred
        nc = len(model.names)
        model = YOLO(args.cfg)
        model.model = type(model.model)(model.model.yaml, nc=nc, verbose=False)  # init's classes: class layer transfers
        model.load(init)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
                workers=args.workers, project=args.out, name=args.name, exist_ok=True, fliplr=0.0, mosaic=1.0,
                close_mosaic=3, cos_lr=True, lr0=args.lr0, amp=True, patience=0, plots=False, seed=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--init", required=True, help="initial detector checkpoint (e.g. FAH shared/detector.pt)")
    p.add_argument("--cfg", default="", help="model yaml (weights from --init where shapes match)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="detector")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, required=True, help="training batch size; choose for your setup")
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--device", required=True, help="training device selection for your setup")
    p.add_argument("--workers", type=int, default=12)
    main(p.parse_args())
