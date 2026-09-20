import os
import gradio as gr
import time
from datetime import datetime
from core import build_object_from_config_file
import yaml
import shutil


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes}:{seconds}"
    elif minutes > 0:
        return f"{minutes}:{seconds}"
    else:
        return f"{seconds}s"


class StorageManager:
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_session_dir(self):
        """创建新的会话目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def get_output_path(self, session_dir, prefix="result"):
        """获取输出文件路径"""
        return os.path.join(session_dir, f"{prefix}.mp4")


class WanAnimate2App:
    def __init__(self, config_path, output_dir="outputs"):
        self.config = config_path
        self.pipeline = None
        self.storage_manager = StorageManager(output_dir)
        self.setup_distributed()

    def setup_distributed(self):
        self.pipeline = build_object_from_config_file(self.config)

    def process_video(self, refer_img, tpl_video_path, width, height, fps=30, seed=-1,
                      clip_len=65, sample_guide_scale=1.0, step=20,
                      prompt=None, prompt_ref=None):
        # 创建新的会话目录
        session_dir = self.storage_manager.create_session_dir()

        # 保存上传的参考图片
        refer_img_path = os.path.join(session_dir, "reference.png")
        refer_img.save(refer_img_path)

        # 复制模板视频到会话目录
        template_path = os.path.join(session_dir, "template.mp4")
        shutil.copy2(tpl_video_path, template_path)

        # 设置输出路径
        # output_path = self.storage_manager.get_output_path(session_dir)

        try:
            tic = time.time()
            # 调用pipeline进行推理
            result = self.pipeline(
                refer_img_path=refer_img_path,
                tpl_video_path=template_path,
                output_path=session_dir,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
                clip_len=clip_len,
                sample_guide_scale=sample_guide_scale,
                step=step,
                prompt=prompt,
                prompt_ref=prompt_ref
            )

            # 保存处理参数
            params = {
                "width": width,
                "height": height,
                "fps": fps,
                "seed": seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(os.path.join(session_dir, "params.yaml"), "w") as f:
                yaml.dump(params, f)

            toc = time.time()
            print(f"Processing time: {format_time(toc - tic)}")
            return result, f"Processing time: {format_time(toc - tic)}"

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None, f"Error during processing: {str(e)}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--refer-img-file", type=str, default="../examples/demo1/reference.png")
    parser.add_argument("--refer-video-file", type=str, default="../examples/demo1/qmww-CCY-25.mov")

    parser.add_argument("--config", type=str, default="./wan_animate_2.yaml")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip_len", type=int, default=81)
    parser.add_argument("--sample_guide_scale", type=float, default=3.0)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--prompt_ref", type=str, default="人物动作的参考视频")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录路径")
    args = parser.parse_args()

    storage_manager = StorageManager(args.output_dir)
    session_dir = session_dir = storage_manager.create_session_dir()
    pipeline = build_object_from_config_file(args.config)

    tic = time.time()
    result = pipeline(
            refer_img_path=args.refer_img_file,
            tpl_video_path=args.refer_video_file,
            output_path=session_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
            clip_len=args.clip_len,
            sample_guide_scale=args.sample_guide_scale,
            step=args.step,
            prompt=args.prompt,
            prompt_ref=args.prompt_ref)
    

    toc = time.time()
    print(f"Processing time: {format_time(toc - tic)}")
    