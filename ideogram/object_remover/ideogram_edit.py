"""Mask-guided local editing; keeps the existing removal engines independent."""
import math
import uuid

from .ideogram_graph import caption_graph, inpaint_graph
from .private_comfy import PrivateComfy, caption_json
from .regions import prepare_region, png

EDIT_CAPTION_SCHEMA = '''You write Ideogram 4 JSON captions for masked image editing.
Describe the intended AFTER image, not the editing procedure. Preserve unedited objects,
their positions, camera, lighting and material. Never invent decorative objects or text.
Output only one JSON object with exactly these keys in this order:
{"high_level_description":"Short description of the desired result.",
 "compositional_deconstruction":{"background":"The background and lighting.","elements":[]}}
For every remaining discrete foreground object add an object to elements:
{"type":"obj","bbox":[y1,x1,y2,x2],"desc":"Its actual appearance."}
For visible text use {"type":"text","bbox":[y1,x1,y2,x2],"text":"Exact text","desc":"Appearance."}.
Coordinates are integers 0–1000 relative to the supplied crop; bbox is optional when unclear.
elements is always an array of JSON objects, NEVER strings. If no objects remain, use [].
Do not add a "none" element. Do not mention an object that was removed, even as "no object".
Keep captions concise. Use literal Unicode. No commentary, markdown fences or extra keys.'''


class IdeogramEditing:
    def __init__(self):
        self.engine = PrivateComfy()

    def caption(self, region, instruction):
        if not self.engine.caption_model.is_file():
            raise ValueError("Local caption model is missing. Supply caption JSON or configure IDEOGRAM_CAPTION_MODEL.")
        if not instruction.strip() or len(instruction) > 8000:
            raise ValueError("Describe the edit in 1–8000 characters.")
        # Describe the selection separately: a painted overlay can leak into the model's
        # requested output as an actual red object, even when told to ignore it.
        x1, y1, x2, y2 = region.mask.getbbox()
        bbox = [round(y1 * 1000 / region.image.height), round(x1 * 1000 / region.image.width),
                round(y2 * 1000 / region.image.height), round(x2 * 1000 / region.image.width)]
        name = uuid.uuid4().hex + "-caption.png"
        request = (f"Aspect ratio {region.image.width}:{region.image.height}. "
                   f"The editing mask is bounded by [y1,x1,y2,x2]={bbox} in 0–1000 coordinates. "
                   "These coordinates describe the editable area, not an object to include in the caption. "
                   "The image shown is BEFORE editing. Describe ONLY the desired AFTER image. "
                   "Do not describe the mask, selection, edit operation, or a removed object. " + instruction)
        try:
            for attempt in range(2):
                outputs = self.engine.run(caption_graph(name, self.engine.caption_model.name, request, EDIT_CAPTION_SCHEMA),
                                          {name: png(region.image)})
                text = outputs.get("4", {}).get("text", [])
                if isinstance(text, list):
                    text = "\n".join(text)
                if not text:
                    raise RuntimeError("Local caption model returned no text")
                try:
                    return caption_json(text)
                except ValueError as exc:
                    if attempt:
                        raise ValueError(f"Local caption failed validation after one retry: {exc}. "
                                         "Use Review / edit caption JSON to provide a caption manually.") from exc
                    request += ("\nCorrect the JSON format of this draft without changing the intended edit. "
                                f"Validation error: {exc}\nDraft: {text[:16000]}\nReturn the corrected JSON only.")
        finally:
            # Drop the caption process/models before loading both large DiTs.
            self.engine.stop()

    def edit(self, image, mask, *, instruction, caption, resolution, padding, feather, invert,
             steps, seed, guidance, strength):
        if not 4 <= steps <= 60 or not 0 <= seed <= 2**64 - 1:
            raise ValueError("Steps must be 4–60 and seed an unsigned 64-bit integer.")
        if not math.isfinite(guidance) or not 1 <= guidance <= 10:
            raise ValueError("Ideogram guidance must be 1–10.")
        if not math.isfinite(strength) or not .1 <= strength <= 1 or int(steps * strength) < 1:
            raise ValueError("Denoising strength must be .1–1 with at least one sampling step.")
        region = prepare_region(image, mask, resolution=resolution, padding=padding, feather=feather, invert=invert)
        caption = caption_json(caption) if caption.strip() else self.caption(region, instruction)
        token = uuid.uuid4().hex
        source_name, mask_name = token + "-image.png", token + "-mask.png"
        graph = inpaint_graph(source_name, mask_name, caption, *region.image.size, steps, seed, guidance, strength)
        try:
            outputs = self.engine.run(graph, {source_name: png(region.image), mask_name: png(region.mask)})
            return region.composite(self.engine.image_output(outputs))
        finally:
            # Explicitly ephemeral; guarantees VRAM release after errors and successful edits.
            self.engine.stop()
