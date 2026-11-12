import base64
import copy
import json
import math
import os
import socket
import subprocess
import sys
import time
import wave
import zipfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from state import EXPORTS, MODELS, PIN, ROOT, atomic, read, require_models, run_path


def stage(job, text):
    atomic(job / "progress.json", {"stage": text})
    print(text, flush=True)


def command(args, timeout=7200):
    subprocess.run([str(a) for a in args], check=True, timeout=timeout, stdin=subprocess.DEVNULL)


def prepare(job, cfg):
    if cfg["source_url"]:
        url = urlparse(cfg["source_url"])
        if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password:
            raise ValueError("Use an HTTP(S) media, episode or video URL")
        stage(job, "Importing the selected media URL")
        command([sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-progress",
                 "--js-runtimes", f"deno:{ROOT / '.venv/bin/deno'}",
                 "--max-filesize", "2G", "--socket-timeout", "30", "--retries", "2",
                 "-f", "best" if cfg["task"] == "subtitles" else "bestaudio/best",
                 "-o", job / "upload", "--", cfg["source_url"]], timeout=1800)
    source = job / "upload"
    if not source.is_file() or not source.stat().st_size:
        raise ValueError("No readable media input")
    stage(job, "Preparing audio and local playback")
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)], timeout=60))
    streams = probe.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise ValueError("The media has no audio track")
    duration = float(probe.get("format", {}).get("duration", 0) or 0)
    if duration > 3600.05:
        raise ValueError("Choose a recording of at most one hour; split longer recordings explicitly")
    command(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source, "-map", "0:a:0",
             "-t", "3600.1", "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", job / "audio.wav"])
    with wave.open(str(job / "audio.wav")) as sound:
        duration = sound.getnframes() / sound.getframerate()
    if not 0 < duration <= 3600.05:
        raise ValueError("Choose a nonempty recording of at most one hour")
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is not None:
        command(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
                 "-map", f"0:{video['index']}", "-map", "0:a:0", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-movflags", "+faststart", job / "preview.mp4"])
    # The waveform describes the actual 16 kHz signal, not synthetic activity.
    with wave.open(str(job / "audio.wav")) as sound:
        import array
        peaks = []
        step = max(1, math.ceil(sound.getnframes() / 1024))
        while raw := sound.readframes(step):
            samples = array.array("h", raw)
            if sys.byteorder != "little":
                samples.byteswap()
            peaks.append(max(abs(x) for x in samples) / 32768)
    if cfg["task"] == "karaoke":
        stage(job, "Separating vocals · CPU")
        command([ROOT / ".stem-venv/bin/python", ROOT / "stems.py", job])
    atomic(job / "media.json", {"duration": duration, "video": video is not None, "peaks": peaks})


def clock(seconds, separator=","):
    value = max(0, round(seconds * 1000))
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    secs, millis = divmod(value, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def export(job, response):
    choice = response["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    (job / "transcript.txt").write_text(text, encoding="utf-8")
    atomic(job / "response.json", response)
    alignment = response.get("ctc_timestamps") or {}
    words = alignment.get("words") or []
    atomic(job / "words.json", alignment)
    cues, current = [], []
    for word in words:
        if not all(isinstance(word.get(k), (float, int)) and math.isfinite(word[k]) for k in ("start", "end")):
            continue
        if not isinstance(word.get("word"), str) or word["end"] < word["start"]:
            continue
        if current and (word.get("speaker") != current[0].get("speaker")
                        or word["end"] - current[0]["start"] > 6
                        or sum(len(w["word"]) + 1 for w in current) + len(word["word"]) > 80):
            cues.append(current)
            current = []
        current.append(word)
    if current:
        cues.append(current)
    packed, srt, vtt, lrc = [], [], ["WEBVTT\n"], []
    for index, group in enumerate(cues, 1):
        start, end = max(0, group[0]["start"]), max(w["end"] for w in group)
        line = " ".join(w["word"] for w in group)
        speaker = group[0].get("speaker")
        tagged = f"[{speaker}] {line}" if speaker is not None else line
        srt.append(f"{index}\n{clock(start)} --> {clock(end)}\n{tagged}\n")
        vtt.append(f"{clock(start, '.')} --> {clock(end, '.')}\n{tagged}\n")
        lrc.append(f"[{int(start // 60):02}:{start % 60:05.2f}]{line}")
        packed.append(dict(start=start, end=end, speaker=speaker, text=line, words=group))
    if packed:
        for filename, lines in (("transcript.srt", srt), ("transcript.vtt", vtt), ("lyrics.lrc", lrc)):
            (job / filename).write_text("\n".join(lines), encoding="utf-8")
    warnings = []
    if choice.get("finish_reason") == "length":
        warnings.append("Output reached the token limit. The partial transcript is saved; increase the output budget to reprocess.")
    if alignment.get("error"):
        warnings.append(f"Word alignment: {alignment['error']}")
    if read(job / "run.json")["settings"]["timestamps"] and not packed:
        warnings.append("No word timing was returned. Text and the raw response remain available.")
    result = dict(text=text, reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
                  cues=packed, warnings=warnings, finish_reason=choice.get("finish_reason"),
                  usage=response.get("usage"), exports=sorted(n for n in EXPORTS if (job / n).is_file()))
    return result


def infer(job, cfg):
    stage(job, "Loading Nemotron · NVFP4 / FP8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    secret = os.urandom(32).hex()
    args = [str(ROOT / ".venv/bin/vllm"), "serve", str(MODELS / "checkpoint"),
            "--served-model-name", PIN["model"], "--host", "127.0.0.1", "--port", str(port),
            "--api-key", secret, "--trust-remote-code", "--dtype", "bfloat16", "--kv-cache-dtype", "fp8",
            "--gpu-memory-utilization", str(cfg["memory"]), "--max-model-len", str(cfg["context"]),
            "--max-num-batched-tokens", "4096", "--max-num-seqs", "1",
            "--enable-chunked-prefill", "--no-enable-prefix-caching", "--use-replayssm",
            "--mamba-backend", "flashinfer", "--mamba-cache-mode", "none",
            "--mm-processor-cache-gb", "0", "--no-async-scheduling",
            "--limit-mm-per-prompt", '{"audio":{"count":1,"length":57600000}}',
            "--reasoning-parser", "nemotron_v3", "--trust-request-mm-kwargs",
            "--middleware", "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware"]
    if cfg["mtp"]:
        args += ["--speculative-config", '{"method":"mtp","num_speculative_tokens":8}']
    # Inherit the supervised worker's process group, including engine subprocesses.
    engine = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL)
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if engine.poll() is not None:
                raise RuntimeError("Native engine stopped during startup; open the log")
            try:
                with opener.open(Request(base + "/v1/models", headers=headers), timeout=2) as ready:
                    if any(m.get("id") == PIN["model"] for m in json.load(ready).get("data", [])):
                        break
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Native engine startup exceeded 30 minutes")
        history = read(job / "prior.json", []) if cfg["continue_chat"] else []
        if not history:
            history = ([{"role": "system", "content": cfg["system"]}] if cfg["system"].strip() else [])
            history.append({"role": "user", "content": [
                {"type": "text", "text": cfg["prompt"]},
                {"type": "audio_url", "audio_url": {"url": "$audio"}}]})
        else:
            history.append({"role": "user", "content": cfg["prompt"]})
        encoded = base64.b64encode((job / "audio.wav").read_bytes()).decode()
        messages = copy.deepcopy(history)
        for message in messages:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "audio_url" and part["audio_url"]["url"] == "$audio":
                        part["audio_url"]["url"] = "data:audio/wav;base64," + encoded
        payload = dict(model=PIN["model"], messages=messages, stream=False, n=1,
                       temperature=0.6 if cfg["reasoning"] else 0, top_p=0.95 if cfg["reasoning"] else 1,
                       max_tokens=cfg["max_tokens"], seed=0, skip_special_tokens=False,
                       spaces_between_special_tokens=False, chat_template_kwargs={"enable_thinking": cfg["reasoning"]})
        if cfg["reasoning"]:
            payload.update(top_k=-1, thinking_token_budget=17408)
            payload["chat_template_kwargs"]["reasoning_budget"] = 16384
        if cfg["timestamps"]:
            payload["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
        stage(job, "Transcribing and aligning words" if cfg["timestamps"] else "Listening")
        try:
            with opener.open(Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers),
                             timeout=7200) as connection:
                response = json.load(connection)
        except HTTPError as error:
            raise RuntimeError(error.read(16000).decode(errors="replace")) from None
        history.append({"role": "assistant", "content": response["choices"][0]["message"].get("content") or ""})
        atomic(job / "conversation.json", history)
        result = export(job, response)
        with zipfile.ZipFile(job / "result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(EXPORTS - {"result.zip"}):
                if (job / name).is_file():
                    archive.write(job / name, name)
        result["exports"] = sorted(n for n in EXPORTS if (job / n).is_file())
        atomic(job / "result.json", result)
        stage(job, "Saved · releasing model")
    finally:
        engine.terminate()
        try:
            engine.wait(timeout=10)
        except subprocess.TimeoutExpired:
            engine.kill()
            engine.wait()


def main():
    job = run_path(sys.argv[1])
    try:
        require_models()
        cfg = read(job / "run.json")["settings"]
        prepare(job, cfg)
        infer(job, cfg)
    except Exception as error:
        atomic(job / "error.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
