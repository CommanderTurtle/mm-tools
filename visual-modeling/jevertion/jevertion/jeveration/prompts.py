"""How a request enters the backbone: the prompt or instruction in the chat format, and a source image as visual tokens.

    text-to-image    [chat head] "Generate an image: <prompt>" [chat tail]
    with an image    [chat head] [visual tokens of the image] [end of vision] <instruction> [chat tail]
"""
import torch

from . import config

IMAGE_PAD = "<|image_pad|>"
_MARKER = "\u0000PROMPT\u0000"


class PromptTokenizer:
    """a text-to-image prompt -> token ids in the backbone's chat format; the prompt is cut to MAX_PROMPT_TOKENS tokens"""

    def __init__(self, tok, max_prompt_tokens=config.MAX_PROMPT_TOKENS):
        self.tok, self.max_prompt_tokens = tok, max_prompt_tokens
        text = tok.apply_chat_template([{"role": "user", "content": config.T2I_TEMPLATE.format(prompt=_MARKER)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        head, tail = text.split(_MARKER)
        self.head_ids = tok(head, add_special_tokens=False)["input_ids"]
        self.tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
        self.pad_id = tok.pad_token_id

    def encode(self, prompt):
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        return self.head_ids + [int(t) for t in ids[: self.max_prompt_tokens]] + self.tail_ids

    def __call__(self, prompts):
        """list of prompts -> (ids [B, L], mask [B, L]), left padded"""
        rows = [self.encode(p) for p in prompts]
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, n - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, n - len(r):] = 1
        return ids, mask


class SourcePromptTokenizer:
    """the text pieces around an image: head (up to the image), mid (up to the instruction) and tail (after it)"""

    def __init__(self, tok, max_prompt_tokens=config.MAX_PROMPT_TOKENS):
        self.tok, self.max_prompt_tokens = tok, max_prompt_tokens
        text = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _MARKER}]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert text.count(IMAGE_PAD) == 1 and text.count(_MARKER) == 1, f"unexpected chat template: {text!r}"
        head, rest = text.split(IMAGE_PAD)
        mid, tail = rest.split(_MARKER)
        self.head_ids, self.mid_ids, self.tail_ids = (tok(s, add_special_tokens=False)["input_ids"] for s in (head, mid, tail))
        assert len(self.mid_ids) >= 1, "the template has no end-of-vision token after the image"

    def encode(self, instruction, cut=True):
        """token ids of an instruction; cut: to MAX_PROMPT_TOKENS"""
        ids = self.tok(instruction, add_special_tokens=False)["input_ids"]
        return ids[: self.max_prompt_tokens] if cut else ids


def patchify(visual, x):
    """uint8 [B, H, W, 3] -> the vision tower's inputs: patch pixel values in [-1, 1] and grid_thw [B, 3]"""
    vc = visual.config
    p, m, t = vc.patch_size, vc.spatial_merge_size, vc.temporal_patch_size
    b, h, w, _ = x.shape
    assert h % (p * m) == 0 and w % (p * m) == 0, f"image {h}x{w} is not a multiple of {p * m}"
    gh, gw = h // p, w // p
    v = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    v = v.reshape(b, 3, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
    v = v.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(b * gh * gw, 3 * t * p * p)
    return v, torch.tensor([[1, gh, gw]] * b, dtype=torch.long, device=x.device)


@torch.no_grad()
def vision_tokens(visual, x):
    """uint8 [B, H, W, 3] (sides multiples of 32) -> visual tokens [B, gh*gw, hidden] in raster order, and (gh, gw)"""
    pv, thw = patchify(visual, x)
    out = visual(pv.to(visual.dtype), grid_thw=thw, return_dict=True).pooler_output
    cell = visual.config.patch_size * visual.config.spatial_merge_size
    gh, gw = x.shape[1] // cell, x.shape[2] // cell
    return out.view(x.shape[0], gh * gw, -1), (gh, gw)


def build_prefix(lm, ptok, vis, grid, instr_ids):
    """vis [B, gh*gw, hidden] (vision_tokens), grid (gh, gw), instr_ids (token ids per row) -> the input prefix of a request
    with an image: (embeddings [B, L, hidden], mask [B, L], rotary positions [3, B, L]), left padded"""
    emb = lm.embed_tokens
    dev, dt = vis.device, emb.weight.dtype
    tk = lambda ids: emb(torch.tensor(ids, dtype=torch.long, device=dev))      # noqa: E731
    head, mid, tail = tk(ptok.head_ids), tk(ptok.mid_ids), tk(ptok.tail_ids)
    gh, gw = grid
    s = len(ptok.head_ids)
    row = torch.arange(gh, device=dev).repeat_interleave(gw)
    col = torch.arange(gw, device=dev).repeat(gh)
    img_pos = torch.stack([torch.full_like(row, s), s + row, s + col])            # [3, gh*gw]
    after = s + max(gh, gw)                                                        # the next text position after the image
    rows, poss = [], []
    for i in range(vis.shape[0]):
        ins = tk(instr_ids[i]) if len(instr_ids[i]) else head[:0]
        n_after = len(ptok.mid_ids) + len(instr_ids[i]) + len(ptok.tail_ids)
        rows.append(torch.cat([head, vis[i].to(dt), mid, ins, tail], 0))
        txt0 = torch.arange(s, device=dev).view(1, -1).expand(3, -1)
        txt1 = (after + torch.arange(n_after, device=dev)).view(1, -1).expand(3, -1)
        poss.append(torch.cat([txt0, img_pos, txt1], 1))
    length = max(r.shape[0] for r in rows)
    b = len(rows)
    x = torch.zeros(b, length, rows[0].shape[1], dtype=dt, device=dev)
    mask = torch.zeros(b, length, dtype=torch.long, device=dev)
    pos = torch.zeros(3, b, length, dtype=torch.long, device=dev)
    for i, r in enumerate(rows):
        x[i, length - r.shape[0]:] = r
        mask[i, length - r.shape[0]:] = 1
        pos[:, i, length - r.shape[0]:] = poss[i]
    return x, mask, pos
