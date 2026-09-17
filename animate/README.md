

<div align="center">

<h2><i><b>Wan-Animate-2: Pushing the Application
Boundaries of Character Animation Models</b></i></h2>

[***Guangyuan Wang***](https://scholar.google.com/citations?user=OYwm0GsAAAAJ&hl=zh-CN)^∗^, [***Li Hu***](https://scholar.google.com/citations?user=Arz3iGUAAAAJ)^*†^, [***Dechao Meng***]()^∗^, [***Zhongyi Zhang***](https://scholar.google.com/citations?user=dOWmKOUAAAAJ)^∗^, [***Peng Zhang***](https://scholar.google.com/citations?user=QTgxKmkAAAAJ)^∗^, [***Xindi Zhang***]()^∗^, [***Mingyang Huang***](https://scholar.google.com/citations?user=EWb6NW4AAAAJ&hl=en), ***Ruoshi Zhang***, ***Ke Sun***, ***Zhe Zhang***, ***Xingjun Wang***, [***Gang Cheng***](https://scholar.google.com/citations?user=nMBg6S8AAAAJ&hl=en&authuser=1), ***Hai Xu***, ***Bang Zhang***^‡^

<sup>∗</sup>Core Contribution &nbsp; <sup>†</sup>Project leaders &nbsp; <sup>‡</sup>Sponsor

<!-- TODO -->
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/Wan-Video/Wan-Animate-2) &nbsp; [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2608.06009-b31b1b.svg)](https://arxiv.org/pdf/2608.06009) &nbsp; [![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://humanaigc.github.io/wan-animate-2) &nbsp; [![Demo](https://img.shields.io/badge/Demo-ModelScope-FF6A00.svg)](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate)

</div>

## 📝 Introduction

We present Wan-Animate-2, a novel end-to-end character animation framework that directly consumes driving videos in a redesigned Diffusion Transformer, which achieves high-fidelity motion generation and strong identity preservation by eliminating intermediate motion extractors. We further add text-driven viewpoint control to decouple the output camera perspective from the driving video. In addition, we develop Wan-Animate-2-Lite, an efficient variant that reduces inference latency to real-time thresholds for streaming character animation.

<div align="center">
<img width="720" alt="architecture" src="assets/pipeline-v2.png"/>
</div>

Release Notes:
- August 07, 2026: 🎉 We release the **Wan-Animate-2** inference scripts.
- August 07, 2026: 🎉 We release the **Wan-Animate-2 Base** model weights.
- August 07, 2026: 🎉 We release the **Wan-Animate-2 Distillation** model weights.
  
## 📑 Todo List
- Wan-Animate-2 Character Animation
    - [x] Inference code of Wan-Animate-2
    - [x] Checkpoints of Wan-Animate-2
    - [x] Diffusers integration
    - [x] DiffSynth-Studio integration
    - [x] ComfyUI integration
    
    
## 🚀 Quick Start
### Installation
1. Clone this repo:
```bash
git clone --recursive https://github.com/Wan-Video/Wan-Animate-2.git
```
<!-- TODO -->
2. Create python environment and install torch:
```bash
conda create -n wan_animate_2 python==3.11 -y
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```
3. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```
4. Install [`flash_attn`](https://github.com/Dao-AILab/flash-attention):
```bash
pip install flash-attn --no-build-isolation
```
5. Install this repository as a local editable package:
```bash
pip install -e .
```

### Model Download

| Model | Download Links |
| :-----  | :----- |
| **Wan-Animate-2** | 🤗 [HuggingFace]( https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B) 🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-Animate-2-14B) |

Download models using huggingface-cli:
```sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir ./ckpts/
```
Download models using modelscope-cli:
 ```sh
pip install modelscope
modelscope download --model Wan-AI/Wan2.2-Animate-2-14B --local_dir ./ckpts/
```

### Inference
Before inference, use an LLM model (e.g. Qwen3.7-Plus) with the prompt below to get the image caption, then use it as the prompt for Wan-Animate-2.
```
用中文客观描述图片中的内容，包括以下要点：人物外观描述，不描述动作行为。 背景描述，忽略主观评价和情绪推测。 下面给出描述范例，必须遵循这个范式，不要输出额外的符号： 人物外观描述：穿着一件浅蓝色的校服衬衫，领口和袖口有白色边饰。胸前有一个圆形徽章。 背景描述：背景为明亮、整洁的教室或办公室，氛围安静有序。
```
*Note:*
The default settings in this repository are tuned for 8× A800 GPUs and support 720P video generation. We have also tested 480P generation on 2× A800 GPUs.
If your hardware setup is different, please adjust the parallel configs in the YAML files accordingly.
- `Wan-Animate-2 Base`:
  ```bash
  export PYTHONPATH="$(pwd)"
  cd infer
  python wan_animate_2_demo.py \
    --prompt "人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。" \
    --refer-img-file ../examples/demo1/reference.png  \
    --refer-video-file ../examples/demo1/template.mp4 \
    --config ./wan_animate_2.yaml
  ```
- `Wan-Animate-2 Distillation`:
  ```bash
  export PYTHONPATH="$(pwd)"
  cd infer
  python wan_animate_2_demo.py \
    --prompt "人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。" \
    --refer-img-file ../examples/demo1/reference.png  \
    --refer-video-file ../examples/demo1/template.mp4 \
    --config ./wan_animate_2_distillation.yaml \
    --sample_guide_scale 1.0 
    --step 10
  ```
## 🧨 Diffusers Inference

Wan-Animate-2 is supported by the [🤗 diffusers](https://github.com/huggingface/diffusers) library (see [PR #14412](https://github.com/huggingface/diffusers/pull/14412)).

Install diffusers from source (until the next release):
```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install flash-attn --no-build-isolation
```

- `Wan-Animate-2 Base`:
  ```python
  import torch
  from diffusers import WanAnimate2Pipeline
  from diffusers.utils import export_to_video, load_image

  pipe = WanAnimate2Pipeline.from_pretrained(
      "Wan-AI/Wan2.2-Animate-2-14B-Diffusers", torch_dtype=torch.bfloat16
  ).to("cuda")

  output = pipe(
      image=load_image("../examples/demo1/reference.png"),
      driving_video="../examples/demo1/template.mp4",
      prompt="人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。",
      height=800,
      width=640,
      num_inference_steps=40,
  )

  export_to_video(output.frames[0], "output.mp4", fps=24)
  ```

- `Wan-Animate-2 Distillation` (10 steps, no CFG):
  ```python
  pipe = WanAnimate2Pipeline.from_pretrained(
      "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers", torch_dtype=torch.bfloat16
  ).to("cuda")

  output = pipe(
      image=load_image("../examples/demo1/reference.png"),
      driving_video="../examples/demo1/template.mp4",
      prompt="人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。",
      num_inference_steps=10,
      guidance_scale=1.0,        # no classifier-free guidance
      flow_solver="euler",       # Euler scheduler for distilled model
  )

  export_to_video(output.frames[0], "output.mp4", fps=24)
  ```

## 🤖 Gradio Demo
Try the online demo: [ModelScope Studio](https://www.modelscope.cn/studios/Wan-AI/Wan2.2-Animate)

Run locally:
- `Wan-Animate-2 Base`:
  ```bash
  cd infer
  python wan_animate_2_gradio.py
  ```
- `Wan-Animate-2 Distillation`:
  ```bash
  cd infer
  python wan_animate_2_gradio_distillation.py
  ```

## 📜 Citation
 
If you find this work helpful, please consider citing:
 
```BibTeX
@article{wang2026wananimate2,
  title   = {Wan-Animate-2: Real-Time End-to-End Character Animation via Diffusion Transformer},
  author  = {Wang, Guangyuan and Hu, Li and Meng, Dechao and Zhang, Zhongyi and Zhang, Peng and
             Huang, Mingyang and Zhang, Ruoshi and Sun, Ke and Zhang, Zhe and
             Wang, Xingjun and Cheng, Gang and Zhang, Bang},
  journal = {arXiv preprint arXiv:TODO.06009},
  year    = {2026},
  url     = {https://arxiv.org/abs/TODO.06009}
}
```

## ⚖️ License

This project is licensed under the [Apache License 2.0](LICENSE).
