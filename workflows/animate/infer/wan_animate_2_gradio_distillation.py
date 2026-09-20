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


def create_ui(config_path, output_dir="outputs"):
    # 加载配置
    app = WanAnimate2App(config_path, output_dir)

    # 创建Gradio界面
    with gr.Blocks(title="Wan Animate 2") as demo:
        gr.Markdown("# Wan Animate 2")

        with gr.Row():
            with gr.Column():
                refer_img = gr.Image(label="Reference Image", type="pil")
                tpl_video = gr.Video(label="Template Video")

                with gr.Row():
                    width = gr.Number(label="Width", value=720)
                    height = gr.Number(label="Height", value=1280)

                with gr.Row():
                    fps = gr.Number(label="FPS", value=24)
                    seed = gr.Number(label="Seed (-1 for random)", value=123)
                    step = gr.Number(label="step", value=10)

                with gr.Row():
                    clip_len = gr.Number(label="CLIP_LEN", value=81)
                    sample_guide_scale = gr.Number(label="CFG", value=1.0)

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="描述你想要创建的内容...",
                    lines=3,
                    max_lines=10,
                    value="描述你想要创建的内容...",  # 默认值
                    info="描述你想要创建的内容..."
                )

                prompt_ref = gr.Textbox(
                    label="Prompt_ref",
                    placeholder="描述输入视频内容...",
                    lines=3,
                    max_lines=10,
                    value="人物动作的参考视频",  # 默认值
                    info="描述输入视频内容..."
                )

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_video = gr.Video(label="Generated Video")
                output_status = gr.Textbox(label="Status")

        # 设置生成按钮事件
        generate_btn.click(
            fn=app.process_video,
            inputs=[refer_img, tpl_video, width, height, fps, seed, clip_len, sample_guide_scale, 
                    step, prompt, prompt_ref],
            outputs=[output_video, output_status]
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./wan_animate_2_distillation.yaml")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录路径")
    args = parser.parse_args()
    demo = create_ui(args.config, args.output_dir)
    demo.launch(server_name='0.0.0.0', server_port=args.port)
