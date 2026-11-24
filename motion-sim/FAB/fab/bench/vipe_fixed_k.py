"""ViPE 1.2 with fixed pinhole intrinsics (world mode's focal rule: focal = median(ViPE, GeoCalib, MoGe-2), then ViPE
with K fixed). A launcher for ViPE's run.py (cwd = the ViPE checkout, its hydra overrides as arguments) that replaces the
default pipeline's GeoCalib intrinsics step by the given fx, fy, cx, cy (env VIPE_FIXED_K, in the video's pixels). With
pipeline.init.intrinsics=gt ViPE's SLAM keeps them fixed. bench/world_vipe.py --fixed_focal_json runs it.
"""

import os
import runpy
import sys

import torch


def main():
    sys.path.insert(0, os.getcwd())  # the ViPE checkout
    import vipe.pipeline.default as vd
    from vipe.pipeline.processors import TrackAnythingProcessor
    from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor
    from vipe.utils.cameras import CameraType

    K = torch.as_tensor([float(x) for x in os.environ["VIPE_FIXED_K"].split(",")]).float()

    class FixedIntrinsicsProcessor(StreamProcessor):
        def update_attributes(self, previous_attributes):
            return previous_attributes | {FrameAttribute.INTRINSICS}

        def __call__(self, frame_idx, frame):
            frame.intrinsics = K.clone()
            frame.camera_type = CameraType.PINHOLE
            return frame

    def init_processors(self, video_stream):
        assert self.camera_type == CameraType.PINHOLE, "fixed intrinsics: pinhole only"
        procs = [FixedIntrinsicsProcessor()]
        if self.init_cfg.instance is not None:
            procs.append(TrackAnythingProcessor(self.init_cfg.instance.phrases, add_sky=self.init_cfg.instance.add_sky,
                                                sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                                                model_cache=self.model_cache))
        return ProcessedVideoStream(video_stream, procs)

    vd.DefaultAnnotationPipeline._add_init_processors = init_processors
    sys.argv = [os.path.join(os.getcwd(), "run.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
