"""vLLM inference for all text-prompted box and point benchmarks.

Each annotation row is turned into one or more requests: detection with
``--prompt_mode single`` and per-category pointing send one request per
category. Rows ``[start_idx, end_idx)`` are written to ``save_path`` in the
order of the annotation file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.utils.parser import parse_prediction  # noqa: E402
from evaluation.utils.prompt_builder import (  # noqa: E402
    PER_CATEGORY_DATASETS,
    POINTING_BENCHMARKS,
    build_prompt,
    pointing_bench_prompt,
)

SYSTEM_PROMPT = "You are a helpful assistant."
IDENTITY_FIELDS = (
    "sample_id",
    "qa_index",
    "gt_count",
    "is_focused",
    "image_id",
    "benchmark_task",
    "benchmark_language",
    "benchmark_slice",
    "is_negative",
    "query_label",
    "source_line",
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_jsonl_path", required=True)
    parser.add_argument("--image_root_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--eval_type", choices=("box_eval", "point_eval"), default="box_eval")
    parser.add_argument(
        "--prompt_mode",
        choices=("multiple", "single"),
        default="multiple",
        help="single: one request per category on COCO/LVIS/Dense200/VisDrone detection",
    )
    parser.add_argument(
        "--prompt_style",
        choices=("routed", "question"),
        default="routed",
        help="question: send each row's own `question` (PointBench, PixMo-Points-Eval)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_pixels", type=int, default=16 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=2025 * 32 * 32)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="vLLM context (prompt + answer); answers stop early once it is reached",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=64, help="requests per vLLM call")
    return parser.parse_args()


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
        )
        self.llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            tokenizer_mode="slow",
            limit_mm_per_prompt={"image": 10, "video": 10},
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            # Prefix-cache hits change prefill kernel shapes and can flip greedy tokens.
            enable_prefix_caching=False,
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.05,
            skip_special_tokens=False,
            stop=["<|im_end|>"],
        )

    def request(self, image: Image.Image, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # The reported numbers use qwen_vl_utils' default 28-px resize factor.
        images, _ = process_vision_info(messages)
        return {"prompt": text, "multi_modal_data": {"image": images}}

    def generate(self, requests: list[dict]) -> list[str]:
        try:
            outputs = self.llm.generate(requests, sampling_params=self.sampling, use_tqdm=False)
            return [output.outputs[0].text for output in outputs]
        except Exception as exc:  # a corrupted image should not fail the whole batch
            print(f"Warning: batched generation failed ({exc}); retrying one by one", flush=True)
            texts = []
            for request in requests:
                try:
                    output = self.llm.generate([request], sampling_params=self.sampling, use_tqdm=False)
                    texts.append(output[0].outputs[0].text)
                except Exception as item_exc:
                    print(f"Warning: generation failed: {item_exc}", flush=True)
                    texts.append("")
            return texts


def ground_truth(entry: dict, eval_type: str):
    """Pointing is scored against instance masks when the annotation has them."""
    gt = entry.get("gt")
    if eval_type == "point_eval":
        if isinstance(entry.get("gt_mask"), dict) and entry["gt_mask"]:
            return entry["gt_mask"]
        if isinstance(gt, dict) and isinstance(gt.get("gt_mask"), dict) and gt["gt_mask"]:
            return gt["gt_mask"]
    return gt


def plan_queries(entry: dict, args: argparse.Namespace) -> list[tuple[str | None, str]]:
    """``[(category or None, prompt), ...]`` for one annotation row."""
    dataset = entry["dataset_name"]
    pointing = args.eval_type == "point_eval"
    categories = [str(c) for c in (entry.get("categories") or list((entry.get("gt") or {})))]
    if args.prompt_style == "question":
        question = (entry.get("question") or "").strip()
        if dataset in POINTING_BENCHMARKS:
            question = pointing_bench_prompt(question)
        return [(None, question)]
    if dataset in PER_CATEGORY_DATASETS and (pointing or args.prompt_mode == "single"):
        return [(c, build_prompt(dataset, [c], pointing=pointing)) for c in categories]
    return [(None, build_prompt(dataset, categories, pointing=pointing))]


def assemble(entry: dict, args: argparse.Namespace, queries, answers, size) -> dict:
    pointing = args.eval_type == "point_eval"
    per_category = queries and queries[0][0] is not None
    extracted: dict = {}
    if size is not None:
        width, height = size
        for (category, _), answer in zip(queries, answers):
            parsed = parse_prediction(answer, width, height, pointing=pointing)
            if not per_category:
                extracted = parsed
            elif pointing:
                points = [s for shapes in parsed.values() for s in shapes if len(s) == 2]
                extracted.setdefault(category, []).extend(points)
            else:
                extracted[category] = [s for shapes in parsed.values() for s in shapes]

    if per_category and not pointing:
        question = {c: q for c, q in queries}
        raw = {c: a for (c, _), a in zip(queries, answers)}
    elif per_category:
        question = " | ".join(q for _, q in queries)
        raw = "\n".join(answers)
    else:
        question = queries[0][1] if queries else ""
        raw = answers[0] if answers else ""

    default_task = "pointing" if pointing else "referring_object_detection"
    prediction = {
        "image_path": entry["image_path"],
        "extracted_predictions": extracted,
        "gt": ground_truth(entry, args.eval_type),
        "question": question,
        "dataset_name": entry["dataset_name"],
        "raw_response": raw,
        "task_name": entry.get("task_name") or default_task,
    }
    gt = entry.get("gt")
    for key in IDENTITY_FIELDS:
        if entry.get(key) is not None:
            prediction[key] = entry[key]
        elif isinstance(gt, dict) and gt.get(key) is not None:
            prediction[key] = gt[key]
    return prediction


def load_image(path: str):
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        print(f"Warning: cannot read {path}: {exc}", flush=True)
        return None


def main() -> None:
    args = get_args()
    with open(args.test_jsonl_path, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    end = len(entries) if args.end_idx < 0 else min(args.end_idx, len(entries))
    entries = entries[args.start_idx : end]
    engine = Engine(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as output:
        cursor = 0
        progress = tqdm(total=len(entries), desc=os.path.basename(args.test_jsonl_path))
        while cursor < len(entries):
            # Fill one vLLM call with whole rows so per-category answers stay together.
            batch, requests = [], []
            while cursor < len(entries) and (not batch or len(requests) < args.batch_size):
                entry = entries[cursor]
                cursor += 1
                queries = plan_queries(entry, args)
                image = load_image(os.path.join(args.image_root_dir, entry["image_path"]))
                if image is not None:
                    requests.extend(engine.request(image, prompt) for _, prompt in queries)
                batch.append((entry, queries, image.size if image is not None else None))
            answers = engine.generate(requests) if requests else []
            offset = 0
            for entry, queries, size in batch:
                count = len(queries) if size is not None else 0
                row_answers = answers[offset : offset + count] or [""] * len(queries)
                offset += count
                prediction = assemble(entry, args, queries, row_answers, size)
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            progress.update(len(batch))
        progress.close()


if __name__ == "__main__":
    main()
