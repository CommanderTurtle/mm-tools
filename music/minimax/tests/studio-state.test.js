import {test, expect} from "bun:test";
import {readFileSync} from "node:fs";
import {readControls, validateControls, parseStudioState, STATE_FORMAT, safeSourceUrl, validateGuideResult} from "../local_app/web/studio-state.js";

const controls = [
  {id: "lyrics", type: "textarea", value: "original"},
  {id: "guideMode", type: "select-one", value: "brief", options: [{value: "brief"}, {value: "keep_lyrics"}, {value: "song"}, {value: "ask"}]},
  {id: "guideWebSearch", type: "checkbox", checked: false},
  {id: "steps", type: "number", value: "30", min: "1", max: "100"},
];
const snapshot = () => ({
  format: STATE_FORMAT, version: 1, session: {version: 1, takes: []},
  ui: {controls: readControls(controls), guide_result: null},
});

test("state roundtrip preserves lyrics and checkbox types", () => {
  const saved = snapshot();
  saved.ui.controls.lyrics = "  [Verse]\r\n中文  \n";
  saved.ui.controls.guideWebSearch = true;
  const parsed = parseStudioState(JSON.stringify(saved), controls);
  expect(parsed.ui.controls).toEqual(saved.ui.controls);
});
test("unknown controls never reach other elements", () => {
  expect(validateControls({...readControls(controls), guideEnabled: true, stateFile: "payload"}, controls))
    .toEqual(readControls(controls));
});
test("rejects corrupt, foreign, future, and mistyped state", () => {
  for (const change of [
    saved => { saved.version = 2; }, saved => { saved.format = "other"; },
    saved => { saved.ui.controls.guideMode = "execute"; },
    saved => { saved.ui.controls.guideWebSearch = "false"; },
    saved => { saved.ui.controls.steps = "Infinity"; },
    saved => { saved.ui.controls.steps = "101"; },
  ]) {
    const saved = snapshot(); change(saved);
    expect(() => parseStudioState(JSON.stringify(saved), controls)).toThrow();
  }
  expect(() => parseStudioState("{", controls)).toThrow();
});
test("imported sources are links, not executable markup", () => {
  expect(safeSourceUrl("javascript:alert(1)")).toBe("");
  expect(safeSourceUrl("https://user:pass@example.org/")).toBe("");
  const result = validateGuideResult({mode: "ask", text: "<script>data only</script>", sources: [
    {url: "javascript:alert(1)"}, {url: "https://example.org/song", title: "<b>Credits</b>"},
  ]});
  expect(result.sources).toHaveLength(1);
  expect(result.text).toContain("<script>");
});
test("all fixed UI references exist; no append or duplicated song controls remain", () => {
  const html = readFileSync(new URL("../local_app/web/index.html", import.meta.url), "utf8");
  const script = readFileSync(new URL("../local_app/web/app.js", import.meta.url), "utf8");
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  expect(new Set(ids).size).toBe(ids.length);
  for (const match of script.matchAll(/\$\("([^"]+)"\)/g)) expect(ids).toContain(match[1]);
  for (const obsolete of ["theoryPresets", "appendTheoryBrief", "guideDuration", "guideCfg", "guideTuningOutput"]) {
    expect(html).not.toContain(`id="${obsolete}"`);
    expect(script).not.toContain(obsolete);
  }
  expect(html).toContain('type="module"');
  expect(html).toContain('value="keep_lyrics"');
});
