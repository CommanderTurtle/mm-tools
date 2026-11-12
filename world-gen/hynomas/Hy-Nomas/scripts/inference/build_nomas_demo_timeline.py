#!/usr/bin/env python3
"""Build a self-contained interactive HTML timeline from a realistic-user run."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


COLORS = {
    "a": "#7c9cff",
    "b": "#57d3a4",
    "c": "#ffb45e",
    "queued": "#ffb45e",
    "running": "#7c9cff",
    "succeeded": "#57d3a4",
    "failed": "#ff6f7d",
}


def clock(timestamp: float | None) -> str:
    if not timestamp:
        return "—"
    return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")


def seconds(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}s"


def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def state_chips(states: list[dict[str, Any]]) -> str:
    compressed: list[dict[str, Any]] = []
    for item in states:
        signature = (
            item.get("status"),
            item.get("queue_position"),
            item.get("worker_id"),
        )
        if compressed and compressed[-1]["signature"] == signature:
            compressed[-1]["end"] = item.get("at")
        else:
            compressed.append(
                {
                    "signature": signature,
                    "start": item.get("at"),
                    "end": item.get("at"),
                    **item,
                }
            )
    return "".join(
        (
            f'<span class="state {esc(item.get("status"))}">'
            f"{clock(item.get('start'))}"
            + (
                f"–{clock(item.get('end'))}"
                if item.get("end") != item.get("start")
                else ""
            )
            + f" · {esc(item.get('status'))}"
            + (
                f" · queue {int(item['queue_position']) + 1}"
                if item.get("status") == "queued"
                and item.get("queue_position") is not None
                else ""
            )
            + "</span>"
        )
        for item in compressed
    )


def video_url(row: dict[str, Any]) -> str:
    value = str(row.get("video_url") or "")
    return Path(value).name if value else ""


def user_column(
    key: str,
    label: str,
    rows: list[dict[str, Any]],
) -> str:
    cards = []
    for row in rows:
        url = video_url(row)
        media = (
            f'<video controls preload="metadata" src="{esc(url)}"></video>'
            if url
            else '<div class="video-placeholder">No video artifact</div>'
        )
        parity = (
            '<span class="pill pass">bit-exact</span>'
            if row.get("bit_exact")
            else (
                '<span class="pill fail">hash mismatch</span>'
                if "bit_exact" in row
                else ""
            )
        )
        cards.append(
            f"""
            <article class="turn-card" data-start="{row.get("submitted_at", 0)}">
              <header>
                <div>
                  <span class="turn-number">Turn {row.get("turn")}</span>
                  <strong>{clock(row.get("submitted_at"))} → {clock(row.get("finished_at"))}</strong>
                </div>
                <div>{parity}<span class="pill">{esc(row.get("status"))}</span></div>
              </header>
              <p class="prompt">{esc(row.get("action"))}</p>
              <div class="states">{state_chips(row.get("states") or [])}</div>
              <dl>
                <div><dt>Requested</dt><dd>{seconds(row.get("requested_seconds"))}</dd></div>
                <div><dt>Accumulated video</dt><dd>{seconds(row.get("duration_seconds"))}</dd></div>
                <div><dt>Generation</dt><dd>{seconds(row.get("generation_seconds"))}</dd></div>
                <div><dt>Worker</dt><dd>{esc(row.get("worker_id"))}</dd></div>
              </dl>
              <details>
                <summary>Hashes and identifiers</summary>
                <code>job: {esc(row.get("job_id"))}</code>
                <code>latent: {esc(row.get("latent_sha256"))}</code>
                <code>baseline: {esc(row.get("baseline_sha256"))}</code>
              </details>
              {media}
            </article>
            """
        )
    subject = esc(rows[0].get("subject") if rows else "")
    return f"""
      <section class="user-column user-{key}">
        <div class="user-heading">
          <span class="avatar">{label}</span>
          <div><h2>User {label}</h2><p>{subject}</p></div>
        </div>
        {"".join(cards)}
      </section>
    """


def assertion_grid(assertions: dict[str, Any]) -> str:
    return "".join(
        f"""
        <div class="assertion {"pass" if value else "fail"}">
          <span>{"✓" if value else "✕"}</span>
          <p>{esc(name.replace("_", " "))}</p>
        </div>
        """
        for name, value in assertions.items()
    )


def build(report: dict[str, Any]) -> str:
    started = float(report.get("started_at") or 0)
    finished = float(report.get("finished_at") or started)
    duration = max(1.0, finished - started)
    samples = report.get("samples") or []
    max_memory = 0.0
    max_power = 0.0
    for sample in samples:
        resources = sample.get("resources") or {}
        max_memory = max(
            max_memory,
            float(resources.get("max_gpu_memory_ratio") or 0) * 100,
        )
        max_power = max(
            max_power,
            float(resources.get("total_power_draw_watts") or 0),
        )
    third = report.get("third_user") or {}
    queued = third.get("observed_before_release") or []
    third_states = state_chips(queued + (third.get("states_after_release") or []))
    third_result = third.get("result") or {}
    third_payload = third_result.get("result") or {}
    raw_third_video = str(third_payload.get("video_url") or "")
    third_video = Path(raw_third_video).name if raw_third_video else ""
    third_media = (
        f'<video controls preload="metadata" src="{esc(third_video)}"></video>'
        if third_video
        else ""
    )
    assertions = report.get("assertions") or {}
    passed = bool(report.get("passed"))
    embedded = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hy-Nomas Demo · Concurrent User Validation</title>
<style>
:root {{ color-scheme: dark; --bg:#080b12; --panel:#111725; --line:#263149; --muted:#94a2bb; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:radial-gradient(circle at 20% 0,#17233e 0,transparent 34%),var(--bg); color:#eef3ff; font:14px/1.5 Inter,system-ui,sans-serif }}
main {{ width:min(1500px,96vw); margin:auto; padding:34px 0 80px }}
.hero,.summary,.third-user,.assertions {{ background:rgba(17,23,37,.92); border:1px solid var(--line); border-radius:20px; padding:24px; margin-bottom:22px; box-shadow:0 20px 70px #0008 }}
h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,54px) }}
h2,p {{ margin:0 }} .muted,.hero p,.user-heading p {{ color:var(--muted) }}
.badge {{ display:inline-flex; padding:6px 10px; border-radius:99px; background:{"#173e31" if passed else "#49242a"}; color:{"#79edbc" if passed else "#ff8e99"}; margin-bottom:12px; font-weight:700 }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-top:22px }}
.metric {{ background:#0b101b; border:1px solid var(--line); padding:14px; border-radius:14px }} .metric strong {{ display:block; font-size:22px }} .metric span {{ color:var(--muted) }}
.timeline-controls {{ display:flex; align-items:center; gap:12px; margin:20px 0 }}
input[type=range] {{ flex:1 }} #clock {{ width:90px; font-variant-numeric:tabular-nums }}
.columns {{ display:grid; grid-template-columns:1fr 1fr; gap:20px }}
.user-column {{ min-width:0 }} .user-heading {{ display:flex; gap:13px; align-items:center; margin:0 0 14px }}
.avatar {{ width:44px; height:44px; display:grid; place-items:center; border-radius:50%; color:#081019; font-weight:900 }}
.user-a .avatar {{ background:{COLORS["a"]} }} .user-b .avatar {{ background:{COLORS["b"]} }}
.turn-card {{ background:rgba(17,23,37,.94); border:1px solid var(--line); border-radius:18px; padding:18px; margin-bottom:16px; transition:.25s }}
.turn-card.future {{ opacity:.18; filter:saturate(.4) }} .turn-card.active {{ border-color:#7c9cff; box-shadow:0 0 0 1px #7c9cff55,0 18px 50px #0008 }}
.turn-card header {{ display:flex; justify-content:space-between; gap:12px }} .turn-number {{ color:var(--muted); margin-right:10px }}
.prompt {{ font-size:16px; margin:14px 0 }} .states {{ display:flex; flex-wrap:wrap; gap:7px }}
.state,.pill {{ display:inline-flex; padding:4px 8px; border-radius:99px; background:#222b3d; font-size:12px }}
.state.queued {{ color:#ffd08c }} .state.running {{ color:#aac0ff }} .state.succeeded,.pill.pass {{ color:#75e8b7 }} .pill.fail,.state.failed {{ color:#ff8e99 }}
dl {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:14px 0 }} dl div {{ background:#0b101b; border-radius:10px; padding:9px }} dt {{ color:var(--muted); font-size:11px }} dd {{ margin:2px 0 0; overflow-wrap:anywhere }}
details code {{ display:block; color:#aebbd2; overflow-wrap:anywhere }} video {{ width:100%; border-radius:12px; background:#000; margin-top:14px }}
.assertions {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px }} .assertion {{ display:flex; gap:10px; align-items:center; background:#0b101b; border-radius:12px; padding:12px }} .assertion.pass span {{ color:#75e8b7 }} .assertion.fail span {{ color:#ff8e99 }}
@media(max-width:900px) {{ .columns,.metrics,.assertions {{ grid-template-columns:1fr }} dl {{ grid-template-columns:1fr 1fr }} }}
</style>
</head>
<body><main>
<section class="hero">
  <span class="badge">{"PASSED" if passed else "VALIDATION IN PROGRESS / FAILED"}</span>
  <h1>Hy-Nomas Demo Concurrent User Timeline</h1>
  <p>{clock(started)} 开始，{clock(finished)} 结束；真实模拟不同到达时间、不同 turn 长度和第三用户排队。</p>
  <div class="metrics">
    <div class="metric"><strong>{seconds(duration)}</strong><span>Wall time</span></div>
    <div class="metric"><strong>{len(report.get("users", {}).get("a", []))}</strong><span>User A turns</span></div>
    <div class="metric"><strong>{len(report.get("users", {}).get("b", []))}</strong><span>User B turns</span></div>
    <div class="metric"><strong>{max_memory:.1f}%</strong><span>Peak GPU memory</span></div>
    <div class="metric"><strong>{max_power:.0f} W</strong><span>Peak total power</span></div>
  </div>
  <div class="timeline-controls"><button id="play">▶ Play timeline</button><input id="scrub" type="range" min="0" max="{duration}" step="1" value="{duration}"><output id="clock">{clock(finished)}</output></div>
</section>
<section class="assertions">{assertion_grid(assertions)}</section>
<div class="columns">
  {user_column("a", "A", report.get("users", {}).get("a", []))}
  {user_column("b", "B", report.get("users", {}).get("b", []))}
</div>
<section class="third-user">
  <h2>User C · Queue validation</h2>
  <p class="prompt">A chef plates a bowl of noodles while both long-session workers are reserved.</p>
  <div class="states">{third_states}</div>
  {third_media}
</section>
</main>
<script id="report-data" type="application/json">{embedded}</script>
<script>
const start={started}, duration={duration};
const scrub=document.querySelector('#scrub'), out=document.querySelector('#clock'), play=document.querySelector('#play');
let timer=null;
function fmt(ts){{return new Date(ts*1000).toLocaleTimeString('zh-CN',{{hour12:false}})}}
function render(){{
  const now=start+Number(scrub.value); out.value=fmt(now);
  document.querySelectorAll('.turn-card').forEach(card=>{{
    const t=Number(card.dataset.start||0); card.classList.toggle('future',t>now); card.classList.toggle('active',Math.abs(t-now)<8);
  }});
}}
scrub.addEventListener('input',render);
play.addEventListener('click',()=>{{
  if(timer){{clearInterval(timer);timer=null;play.textContent='▶ Play timeline';return}}
  if(Number(scrub.value)>=duration) scrub.value=0;
  play.textContent='⏸ Pause';
  timer=setInterval(()=>{{scrub.value=Math.min(duration,Number(scrub.value)+2);render();if(Number(scrub.value)>=duration){{clearInterval(timer);timer=null;play.textContent='▶ Play timeline'}}}},120);
}});
render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).expanduser().read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
