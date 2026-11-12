function seconds(value) {
  const parts =
    typeof value === "number"
      ? [value]
      : String(value).trim().split(":").map(Number);
  if (
    !parts.length ||
    parts.length > 3 ||
    parts.some((v) => !Number.isFinite(v) || v < 0)
  )
    throw Error("Segment timestamps must be seconds or HH:MM:SS");
  return parts.reduce((sum, part) => sum * 60 + part, 0);
}

export function structuredPrompt(raw, fps = 24) {
  if (!raw.trim()) return null;
  const parsed = JSON.parse(raw),
    item = Array.isArray(parsed) ? parsed[0] : parsed;
  if (!item || !Array.isArray(item.segments) || !item.segments.length)
    throw Error("Structured JSON needs a non-empty segments array");
  const boundaries = [0];
  let duration = 0;
  const prompts = item.segments.map((segment, index) => {
    const actions = Array.isArray(segment.contained_actions)
      ? segment.contained_actions
      : segment.contained_actions
        ? [segment.contained_actions]
        : [];
    const starts = actions
      .filter((a) => a.start_time !== undefined)
      .map((a) => seconds(a.start_time));
    const ends = actions
      .filter((a) => a.end_time !== undefined)
      .map((a) => seconds(a.end_time));
    const length =
      segment.end_time !== undefined
        ? seconds(segment.end_time) - seconds(segment.start_time ?? 0)
        : Math.max(...ends) - (starts.length ? Math.min(...starts) : 0);
    if (!Number.isFinite(length) || length <= 0)
      throw Error("Each segment needs a positive duration");
    duration += length;
    boundaries.push(
      Math.max(boundaries.at(-1) + 1, Math.floor(duration * fps)),
    );
    return JSON.stringify(
      index ? { segments: [segment] } : { ...item, segments: [segment] },
    );
  });
  return {
    prompt: prompts.join("|||"),
    num_frames: boundaries.at(-1),
    segment_boundaries: boundaries,
  };
}
