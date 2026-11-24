"""Adapter: the SAM 3D Body hand branch, off the shelf (the "SAM 3D Body hand" rows of the paper tables and the
SAM 3D Body panels of the FAH comparison figure) -> benchmark prediction format.

The released SAM 3D Body network (facebook/sam-3d-body-dinov3 `model.ckpt`: DINOv3-H+ encoder + the hand decoder and
heads; gated, SAM License) runs through the FAH pipeline code with its own weights and nothing of FAH's: no fine-tuned
heads, no mesh fit, no fingertip step, no video refiner. Settings of the paper rows:
  - detector: WiLoR's YOLO hand detector (`detector.pt` from rolpotamias/WiLoR, through ultralytics), confidence 0.3,
    NMS IoU 0.5, 512 px letterbox (fah.detector.WiLoRHandDetector defaults; the `eval` extra installs ultralytics);
  - crops: the hand decoder's native 512 px, box padding 0.9, left hands mirrored (as in SAM 3D Body);
  - outputs: the hand decoder's MHR hand mesh and joints (MANO joints through the MHR -> MANO regressor);
  - precision: bfloat16 encoder with the pose network in one CUDA graph (`cuda_graph=False`: the eager timing).

    python -m bench.run --dataset hot3d --split bench/splits/hot3d_bench.json --data_root $DATA_ROOT/hot3d \
        --method sam3db_hand --method_args '{"checkpoint": "<dir>/model.ckpt"}' --run_name sam3db-hand --out_root $RESULTS
"""
from bench.methods.fah import FAHMethod

PAPER_ARGS = dict(dtype="bfloat16", cuda_graph=True, detector="wilor", det_conf=0.3, bbox_padding=0.9, mesh_fit=0,
                  tip_extend=0.0)  # crop size: the checkpoint config's 512 px


class SAM3DBodyHandMethod(FAHMethod):
    name = "sam3db-hand"

    def __init__(self, checkpoint="checkpoints/sam-3d-body-dinov3/model.ckpt", **kwargs):
        args = dict(PAPER_ARGS, **kwargs)
        for k in ("finetune_ckpt", "student", "pose_trt", "contact_ckpt", "contact_video", "refiner"):
            if args.get(k):
                raise ValueError(f"sam3db_hand runs the released SAM 3D Body weights only; got {k}={args[k]!r}")
        args.update(base_weights=True, parts=False)
        super().__init__(checkpoint=checkpoint, **args)
        self.name = "sam3db-hand" + ("" if args["cuda_graph"] else "-eager")
