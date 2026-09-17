export const STATE_FORMAT = "mm-tools.minimax-session";
export const MAX_STATE_BYTES = 8 * 1024 * 1024;
export const GUIDE_MODES = ["brief", "keep_lyrics", "song", "ask"];

export function readControls(controls) {
  return Object.fromEntries(controls.map((control) => [
    control.id, control.type === "checkbox" ? control.checked : control.value,
  ]));
}

export function validateControls(values, controls) {
  if (!values || typeof values !== "object" || Array.isArray(values)) throw new Error("Invalid saved controls.");
  const clean = {};
  for (const control of controls) {
    if (!Object.hasOwn(values, control.id)) continue;
    const value = values[control.id];
    if (control.type === "checkbox") {
      if (typeof value !== "boolean") throw new Error(`Invalid setting: ${control.id}`);
    } else {
      if (typeof value !== "string" || value.length > 50000) throw new Error(`Invalid setting: ${control.id}`);
      if (control.options && !Array.from(control.options).some((option) => option.value === value)) {
        throw new Error(`Unknown choice for ${control.id}`);
      }
      if (control.type === "number" || control.type === "range") {
        const number = Number(value);
        if (value && (!Number.isFinite(number)
          || (control.min !== "" && number < Number(control.min))
          || (control.max !== "" && number > Number(control.max)))) {
          throw new Error(`Invalid number for ${control.id}`);
        }
      }
    }
    clean[control.id] = value;
  }
  return clean;
}

export function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : "";
  } catch (_) { return ""; }
}

export function validateGuideResult(result) {
  if (result == null) return null;
  if (typeof result !== "object" || typeof result.text !== "string" || result.text.length > 2000000) {
    throw new Error("Invalid saved writing result.");
  }
  const sections = {};
  for (const name of ["Global Metadata", "Vocal Details", "Arrangement", "Lyrics"]) {
    const text = result.sections?.[name];
    if (text !== undefined) {
      if (typeof text !== "string" || text.length > 100000) throw new Error("Invalid saved writing section.");
      sections[name] = text;
    }
  }
  const sources = (Array.isArray(result.sources) ? result.sources : []).slice(0, 3).flatMap((source) => {
    if (!source || !safeSourceUrl(source.url)) return [];
    return [{
      url: safeSourceUrl(source.url),
      title: typeof source.title === "string" ? source.title.slice(0, 250) : source.url,
      scraped: source.scraped === true,
    }];
  });
  return {text: result.text, mode: GUIDE_MODES.includes(result.mode) ? result.mode : "brief", sections, sources};
}

export function parseStudioState(text, controls) {
  const saved = JSON.parse(text);
  if (saved?.format !== STATE_FORMAT || saved.version !== 1 || saved.session?.version !== 1
      || !Array.isArray(saved.session.takes) || saved.session.takes.length > 500 || !saved.ui) {
    throw new Error("Not a supported MiniMax Studio state file.");
  }
  return {
    session: saved.session,
    ui: {
      controls: validateControls(saved.ui.controls, controls),
      active_workspace: saved.ui.active_workspace === "promptGuide" ? "promptGuide" : "songStudio",
      selected_job_id: /^[a-f0-9]{16}$/.test(saved.ui.selected_job_id) ? saved.ui.selected_job_id : "",
      guide_result: validateGuideResult(saved.ui.guide_result),
    },
  };
}
