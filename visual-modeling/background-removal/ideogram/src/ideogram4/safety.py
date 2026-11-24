from __future__ import annotations

import io

import requests

HIVE_TEXT_MODERATION_URL = "https://api.thehive.ai/api/v3/hive/text-moderation"
HIVE_VISUAL_MODERATION_URL = "https://api.thehive.ai/api/v3/hive/visual-moderation"

# Visual model returns floats in [0, 1]; flag a class when its value is >= this.
HIVE_VISUAL_THRESHOLD = 0.9
# Text model returns integer severities (0 = none, 1+ = present). Flag at >= 1.
HIVE_TEXT_THRESHOLD = 1

# Visual classes we treat as safety signals.
_VISUAL_SAFETY_CLASSES = {
  "general_nsfw",
  "general_suggestive",
  "yes_sexual_activity",
  "yes_sexual_intent",
  "yes_realistic_nsfw",
  "very_bloody",
  "yes_animal_abuse",
  "yes_self_harm",
  "yes_confederate",
  "yes_kkk",
  "yes_nazi",
}

# Text classes we treat as safety signals.
_TEXT_SAFETY_CLASSES = {
  "bullying",
  "child_exploitation",
  "child_safety",
  "drugs",
  "hate",
  "self_harm",
  "self_harm_intent",
  "sexual",
  "sexual_description",
  "violence",
  "violent_description",
}


def _iter_classes(payload: dict):
  for output in payload.get("output", []):
    yield from output.get("classes", [])


def moderate_prompt(
  prompt: str,
  api_key: str,
  threshold: float = HIVE_TEXT_THRESHOLD,
) -> list[tuple[str, float]]:
  """Run Hive Text Moderation over `prompt`.

  Returns (class_name, value) tuples for safety-relevant categories that
  scored at or above `threshold`. Text model values are integer severities
  (typically 0-3), so the default threshold of 1 flags any detection.
  """
  resp = requests.post(
    HIVE_TEXT_MODERATION_URL,
    headers={"Authorization": f"Bearer {api_key}"},
    json={"input": [{"text": prompt}]},
    timeout=30,
  )
  resp.raise_for_status()
  return [
    (cls["class"], float(cls["value"]))
    for cls in _iter_classes(resp.json())
    if float(cls["value"]) >= threshold and cls["class"] in _TEXT_SAFETY_CLASSES
  ]


def moderate_image(
  image,
  api_key: str,
  threshold: float = HIVE_VISUAL_THRESHOLD,
) -> list[tuple[str, float]]:
  """Run Hive Visual Content Moderation over a PIL `image`.

  Returns (class_name, value) tuples for categories that scored at or above
  `threshold`. Visual model values are floats in [0, 1].
  """
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  buf.seek(0)
  resp = requests.post(
    HIVE_VISUAL_MODERATION_URL,
    headers={"Authorization": f"Bearer {api_key}"},
    files={"media": ("image.png", buf, "image/png")},
    timeout=60,
  )
  resp.raise_for_status()
  return [
    (cls["class"], float(cls["value"]))
    for cls in _iter_classes(resp.json())
    if float(cls["value"]) >= threshold and cls["class"] in _VISUAL_SAFETY_CLASSES
  ]
