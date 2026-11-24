from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path


import requests

from ideogram4.caption_verifier import CaptionVerifier


# Directory holding the system-prompt text files shipped with the package.
SYSTEM_PROMPT_DIR = Path(__file__).resolve().parent / "magic_prompt_system_prompts"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"


class MagicPrompt(ABC):
  """A magic-prompt configuration: rewrites a plain prompt into a caption."""

  @abstractmethod
  def expand(self, prompt: str, aspect_ratio: str = "1:1") -> str:
    """Rewrite ``prompt`` into the structured caption JSON string.

    Args:
      prompt: The user's plain-language idea.
      aspect_ratio: Target image aspect ratio as ``"W:H"`` (e.g. ``"16:9"``).

    Returns:
      The structured caption, expected to be a single-line minified JSON object
      matching the caption schema (validate with ``CaptionVerifier``).
    """


# --------------------------------------------------------------------------- #
# Shared helpers (not part of the MagicPrompt interface; subclasses call these).
# --------------------------------------------------------------------------- #


def aspect_ratio_from_size(width: int, height: int) -> str:
  """Reduce a pixel ``width``x``height`` to a ``"W:H"`` aspect-ratio string."""
  divisor = math.gcd(width, height) or 1
  return f"{width // divisor}:{height // divisor}"


@lru_cache(maxsize=None)
def _load_sections(filename: str) -> dict[str, str]:
  """Parse a system-prompt file into its ``[SECTION]`` blocks.

  Files use ``[NAME]`` markers alone on a line (``[META]``, ``[SYSTEM]``,
  ``[USER]``). Returns a mapping of lower-cased section name to its text body
  with surrounding whitespace stripped. Cached so a file is read at most once.
  """
  raw = (SYSTEM_PROMPT_DIR / filename).read_text(encoding="utf-8")
  sections: dict[str, str] = {}
  current: str | None = None
  lines: list[str] = []
  for line in raw.splitlines():
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
      if current is not None:
        sections[current] = "\n".join(lines).strip()
      current = stripped[1:-1].strip().lower()
      lines = []
    else:
      lines.append(line)
  if current is not None:
    sections[current] = "\n".join(lines).strip()
  if "system" not in sections:
    raise ValueError(f"{filename} has no [SYSTEM] section")
  return sections


def build_messages(
  system_prompt_file: str, prompt: str, aspect_ratio: str
) -> list[dict]:
  """Build the chat ``messages`` list from a system-prompt file.

  The system message is the file's ``[SYSTEM]`` block. The user message comes
  from its ``[USER]`` template, substituting ``{{aspect_ratio}}`` and
  ``{{original_prompt}}``. If the template has no ``{{original_prompt}}``
  placeholder (or there is no ``[USER]`` block) the prompt is appended after a
  default framing line instead.
  """
  sections = _load_sections(system_prompt_file)
  template = sections.get("user")
  if template is None:
    template = "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height)."
  user = template.replace("{{aspect_ratio}}", aspect_ratio)
  if "{{original_prompt}}" in user:
    user = user.replace("{{original_prompt}}", prompt)
  else:
    user = f"{user}\n\n{prompt}"
  return [
    {"role": "system", "content": sections["system"]},
    {"role": "user", "content": user},
  ]


def _strip_code_fences(text: str) -> str:
  """Drop a surrounding ```` ```json ... ``` ```` fence if a model adds one."""
  text = text.strip()
  if not text.startswith("```"):
    return text
  lines = text.splitlines()
  if lines and lines[0].startswith("```"):
    lines = lines[1:]
  if lines and lines[-1].strip() == "```":
    lines = lines[:-1]
  return "\n".join(lines).strip()


def openrouter_chat(
  model: str,
  messages: list[dict],
  api_key: str | None,
  *,
  temperature: float | None = 1.0,
  max_tokens: int = 16384,
  extra_body: dict | None = None,
  timeout: float = 120.0,
) -> str:
  """Run one chat completion through OpenRouter and return its text content.

  OpenRouter's API is OpenAI-compatible, so ``model`` is an OpenRouter slug
  (https://openrouter.ai/models). ``extra_body`` is merged into the request for
  provider-specific knobs. The returned text has markdown code fences stripped.
  """
  if not api_key:
    raise RuntimeError("No API key. Set MAGIC_PROMPT_API_KEY or pass api_key=...")

  body = {"model": model, "messages": messages, "max_tokens": max_tokens}
  if temperature is not None:
    body["temperature"] = temperature
  if extra_body:
    body.update(extra_body)

  resp = requests.post(
    OPENROUTER_URL,
    headers={
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
    },
    json=body,
    timeout=timeout,
  )
  resp.raise_for_status()
  data = resp.json()

  choices = data.get("choices")
  if not choices:
    raise RuntimeError(f"OpenRouter returned no choices: {data}")
  content = choices[0].get("message", {}).get("content")
  if not content:
    raise RuntimeError(f"OpenRouter returned an empty message: {choices[0]}")
  return _strip_code_fences(content)


def _to_ideogram_aspect_ratio(aspect_ratio: str) -> str:
  """Convert a ``"W:H"`` ratio to Ideogram's ``"WxH"`` form (``AUTO`` passes through)."""
  if aspect_ratio.upper() == "AUTO":
    return "AUTO"
  return aspect_ratio.replace(":", "x")


def reorder_caption_keys(caption: dict) -> dict:
  """Reorder a caption's object keys to the canonical schema order in place.

  JSON object key order is semantically irrelevant, but ``CaptionVerifier``
  enforces a canonical order (e.g. elements must be ``type`` before ``desc``).
  The hosted magic-prompt API can return keys in a different order, so we
  reorder ``style_description``, ``compositional_deconstruction``, and each
  element to match. Unknown keys are kept, appended after the known ones.
  """

  verifier = CaptionVerifier()

  def _ordered(d: dict, order) -> dict:
    known = [k for k in order if k in d]
    extra = [k for k in d if k not in order]
    return {k: d[k] for k in (*known, *extra)}

  if not isinstance(caption, dict):
    return caption

  sd = caption.get("style_description")
  if isinstance(sd, dict):
    try:
      caption["style_description"] = _ordered(
        sd, verifier._style_description_key_order(sd)
      )
    except ValueError:
      pass  # ambiguous photo/art_style; leave order untouched for the verifier to flag

  cd = caption.get("compositional_deconstruction")
  if isinstance(cd, dict):
    cd = _ordered(cd, verifier.compositional_deconstruction_key_order)
    elements = cd.get("elements")
    if isinstance(elements, list):
      reordered = []
      for element in elements:
        if isinstance(element, dict):
          try:
            element = _ordered(element, verifier._element_key_order(element))
          except ValueError:
            pass  # missing/unknown "type"; leave order for the verifier to flag
        reordered.append(element)
      cd["elements"] = reordered
    caption["compositional_deconstruction"] = cd

  return caption


def ideogram_magic_prompt(
  prompt: str,
  aspect_ratio: str,
  api_key: str | None,
  *,
  timeout: float = 120.0,
) -> str:
  """Expand a plain prompt via Ideogram's hosted magic-prompt API.

  Unlike the OpenRouter-based configurations, this is a managed service that
  performs the prompt expansion server-side, so no system prompt is sent.
  ``aspect_ratio`` is Ideogram's ``"WxH"`` form (or ``"AUTO"``). The endpoint
  returns ``{"aspect_ratio": ..., "json_prompt": {...}}``; we return the
  ``json_prompt`` object as a minified JSON string.
  """
  if not api_key:
    raise RuntimeError("No API key. Set IDEOGRAM_API_KEY or pass api_key=...")

  resp = requests.post(
    IDEOGRAM_MAGIC_PROMPT_URL,
    headers={
      "Api-Key": api_key,
      "Content-Type": "application/json",
    },
    json={"text_prompt": prompt, "aspect_ratio": aspect_ratio},
    timeout=timeout,
  )
  resp.raise_for_status()
  data = resp.json()

  json_prompt = data.get("json_prompt")
  if not json_prompt:
    raise RuntimeError(f"Ideogram API returned no json_prompt: {data}")
  json_prompt = reorder_caption_keys(json_prompt)
  return json.dumps(json_prompt, ensure_ascii=False, separators=(",", ":"))


def strip_aspect_ratio_and_bboxes(caption: str, *, strip_bboxes: bool = True) -> str:
  data = json.loads(caption)
  data.pop("aspect_ratio", None)
  if strip_bboxes:
    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    for element in elements:
      if isinstance(element, dict):
        element.pop("bbox", None)
  return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Concrete configurations. Each subclass pins a model + system-prompt version.
# --------------------------------------------------------------------------- #


class ClaudeSonnetMagicPromptV1(MagicPrompt):
  """Magic prompt v1 on Claude Sonnet 4.6, via OpenRouter."""

  def __init__(
    self,
    api_key: str | None = None,
    *,
    timeout: float = 120.0,
    strip_bboxes: bool = True,
  ) -> None:
    self.api_key = api_key
    self.timeout = timeout
    self.strip_bboxes = strip_bboxes

  def expand(self, prompt: str, aspect_ratio: str = "1:1") -> str:
    messages = build_messages("v1.txt", prompt, aspect_ratio)
    caption = openrouter_chat(
      "anthropic/claude-sonnet-4.6",
      messages,
      self.api_key,
      temperature=1.0,
      extra_body={"reasoning": {"enabled": False}},
      timeout=self.timeout,
    )
    return strip_aspect_ratio_and_bboxes(caption, strip_bboxes=self.strip_bboxes)


class ClaudeOpusMagicPromptV1(MagicPrompt):
  """Magic prompt v1 on Claude Opus 4.8, via OpenRouter."""

  def __init__(
    self,
    api_key: str | None = None,
    *,
    timeout: float = 120.0,
    strip_bboxes: bool = True,
  ) -> None:
    self.api_key = api_key
    self.timeout = timeout
    self.strip_bboxes = strip_bboxes

  def expand(self, prompt: str, aspect_ratio: str = "1:1") -> str:
    messages = build_messages("v1.txt", prompt, aspect_ratio)
    caption = openrouter_chat(
      "anthropic/claude-opus-4.8",
      messages,
      self.api_key,
      temperature=1.0,
      extra_body={"reasoning": {"enabled": False}},
      timeout=self.timeout,
    )
    return strip_aspect_ratio_and_bboxes(caption, strip_bboxes=self.strip_bboxes)


class Ideogram4MagicPromptV1(MagicPrompt):
  """Magic prompt via Ideogram's hosted ideogram-v4 API.

  A free, managed service from Ideogram. The expansion runs server-side, so
  unlike the OpenRouter configurations there is no system prompt to ship; the
  only input is the user's plain prompt. Authenticate with an Ideogram API key
  (``IDEOGRAM_API_KEY``).
  """

  def __init__(
    self,
    api_key: str | None = None,
    *,
    timeout: float = 120.0,
    strip_bboxes: bool = True,
  ) -> None:
    self.api_key = api_key
    self.timeout = timeout
    self.strip_bboxes = strip_bboxes

  def expand(self, prompt: str, aspect_ratio: str = "1:1") -> str:
    caption = ideogram_magic_prompt(
      prompt,
      _to_ideogram_aspect_ratio(aspect_ratio),
      self.api_key,
      timeout=self.timeout,
    )
    return strip_aspect_ratio_and_bboxes(caption, strip_bboxes=self.strip_bboxes)


MAGIC_PROMPTS: dict[str, type[MagicPrompt]] = {
  "claude-sonnet-v1": ClaudeSonnetMagicPromptV1,
  "claude-opus-v1": ClaudeOpusMagicPromptV1,
  "ideogram-4-v1": Ideogram4MagicPromptV1,
}

DEFAULT_MAGIC_PROMPT = "ideogram-4-v1"
