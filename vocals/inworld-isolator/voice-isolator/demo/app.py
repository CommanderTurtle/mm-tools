# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline files and a live microphone, each using its own loaded model.

    pip install -e ".[demo]"
    python demo/app.py
    python demo/app.py --offline-model offline.safetensors \
        --online-model realtime.safetensors
"""

import argparse
from collections import abc
import dataclasses
import functools
import html
import math
import os
import pathlib
import platform
import subprocess
import sys
import threading
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch

import example_audio
from inworld import voice_isolation
from inworld.voice_isolation import registry
import online

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
LOGO = ASSETS.parent.parent / "docs" / "assets" / "inworld-logo-black.svg"
DEFAULT_LOOKAHEAD = 2
SAMPLE_RATE = 48000
SPEC_HEIGHT = 220
# One Offline request holds the model (and on an Apple GPU the live takes)
# for its whole length: about a quarter of the audio's duration on a GPU.
MAX_OFFLINE_SECONDS = 600
MAX_UPLOAD_SIZE = "300mb"
# Gradio sizes its container to the content, so each tab would set its own
# width; a fixed width keeps both tabs' edges in place. The Offline outputs
# form one grid (players in a row, spectrograms in the next) so the three
# columns line up; narrow screens stack each player above its spectrogram.
CSS = """
.show-api-divider { display: none !important; }
.gradio-container { width: 100% !important; max-width: 1200px !important;
  margin: 0 auto !important; }
#studio-tabs { width: 100%; }
#studio-tabs > div:first-child { margin-bottom: 20px; }
.demo-header { text-align: center; padding: 20px 0 18px; }
.demo-header .logo { width: 170px; margin: 0 auto 14px; }
.demo-header h1 { font-size: 32px; letter-spacing: -1px; margin: 0; }
.demo-header p { opacity: .65; margin: 8px 0 14px; }
.device-badges { display: flex; justify-content: center; gap: 8px;
  flex-wrap: wrap; }
.device-badge { display: inline-flex; align-items: center; gap: 2px 8px;
  flex-wrap: wrap; justify-content: center;
  padding: 5px 14px; border-radius: 999px; font-size: 13px;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary); }
.device-badge > span { white-space: nowrap; }
.device-badge strong { font-weight: 600; }
.device-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.device-gpu .device-dot { background: #22c55e; box-shadow: 0 0 0 4px #22c55e26; }
.device-metal .device-dot { background: #38bdf8; box-shadow: 0 0 0 4px #38bdf826; }
.device-cpu .device-dot { background: #f59e0b; box-shadow: 0 0 0 4px #f59e0b26; }
.device-detail { opacity: .7; }
/* Gradio nests an HTML component's own CSS inside one rule, which drops
   @keyframes there; the live panel's status dot animates with this one. */
@keyframes breathe { 50% { box-shadow: 0 0 0 8px #75f1c508; } }
#offline-grid { display: grid !important; gap: 16px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  grid-template-rows: auto auto; grid-auto-flow: column;
  align-items: stretch !important; }
#offline-grid > .column { display: contents !important; }
#offline-grid .spectrogram .image-frame { width: 100% !important; }
#offline-grid .spectrogram .icon-button-wrapper { display: none !important; }
#offline-grid .spectrogram img { width: 100% !important; height: 100% !important;
  object-fit: fill !important; border-radius: 6px; }
#offline-run { min-height: 48px; }
#live-studio { padding: 0 !important; border: 0 !important;
  background: none !important; box-shadow: none !important; }
#live-studio > .html-container { padding: 0 !important; }
#online-results { margin-top: 16px; gap: 16px; }
@media (max-width: 860px) {
  #offline-grid { grid-template-columns: minmax(0, 1fr);
    grid-template-rows: none; grid-auto-flow: row; }
  #offline-run { position: sticky; bottom: 12px; z-index: 5; }
}
"""


@dataclasses.dataclass
class DemoModels:
    """Separate, fixed-mode models reused for all requests.

    Each model's calls hold its lock. PyTorch's Apple GPU backend cannot run
    work from two threads at once (an offline isolation during a live take
    aborts the process), so two models on that GPU share one lock; an offline
    isolation then holds live takes until it ends.
    """

    offline: voice_isolation.VoiceIsolator
    realtime: voice_isolation.VoiceIsolator
    offline_lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock)
    realtime_lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock)
    offline_name: str = registry.MODE_MODELS["offline"]
    realtime_name: str = registry.MODE_MODELS["streaming"]

    def __post_init__(self):
        if self.offline.device.type == self.realtime.device.type == "mps":
            self.realtime_lock = self.offline_lock


def model_name(spec: str, mode: str) -> str:
    """Names a checkpoint as the tabs show it.

    The Hub repository and registry names give the release name (the mode
    picks the repository's checkpoint); any other checkpoint shows its file
    name.
    """
    if spec == registry.REPO_ID:
        return registry.MODE_MODELS[mode]
    if spec in registry.MODELS or spec in registry.PINNED:
        return spec
    if spec.startswith(registry.HF_SCHEME):
        return registry.parse_hf(spec)[1]
    return pathlib.Path(spec).name


def load_models(offline_model: str = registry.REPO_ID,
                online_model: str = registry.REPO_ID,
                device: str | None = None,
                precision: str = "auto") -> DemoModels:
    """Loads both checkpoints once; an incompatible checkpoint fails startup."""
    offline = voice_isolation.VoiceIsolator.load(offline_model,
                                                 device=device,
                                                 mode="offline",
                                                 precision=precision)
    realtime = voice_isolation.VoiceIsolator.load(online_model,
                                                  device=device,
                                                  mode="streaming",
                                                  precision=precision,
                                                  lookahead_frames=2,
                                                  block_frames=4)
    if offline.mode != "offline" or realtime.mode != "streaming":
        raise ValueError("The demo needs an offline and a realtime checkpoint.")
    if offline.model is realtime.model:
        raise ValueError("Each tab needs a separate model instance.")
    return DemoModels(offline,
                      realtime,
                      offline_name=model_name(offline_model, "offline"),
                      realtime_name=model_name(online_model, "streaming"))


def warm_up(models: DemoModels) -> None:
    """Runs half a second of silence through both models before serving.

    A GPU compiles its kernels on first use: on Apple's GPU the first live
    push takes seconds, long enough for the browser to end the first take as
    falling behind. Short windows make the warm-up stream cross several
    crossfades, so the two-lane path is compiled too.
    """
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    with models.offline_lock:
        models.offline.isolate(silence, SAMPLE_RATE)
    with models.realtime_lock:
        stream = models.realtime.stream(window_frames=8, crossfade_frames=4)
        stream.push(silence)
        stream.flush()


@functools.lru_cache(maxsize=1)
def processor_name() -> str:
    """The processor's name where the system reports one, else its type.

    Read once: on macOS it runs ``sysctl``.
    """
    name = ""
    try:
        if sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True,
                                  text=True,
                                  timeout=2,
                                  check=True).stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                name = next((line.split(":", 1)[1].strip()
                             for line in cpuinfo
                             if line.startswith("model name")), "")
        else:
            name = platform.processor()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or platform.machine() or "unknown processor"


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    """Where inference runs, as the header and speed readouts show it.

    ``kind`` is "gpu" (CUDA), "metal" (Apple's GPU) or "cpu"; ``name`` is
    the short label, for example "Metal GPU"; ``detail`` names the hardware.
    """

    kind: str
    name: str
    detail: str


def describe_device(device: torch.device | str) -> DeviceInfo:
    """Describes an inference device for people rather than for PyTorch."""
    device = torch.device(device)
    if device.type == "cuda":
        return DeviceInfo("gpu", "CUDA GPU", torch.cuda.get_device_name(device))
    if device.type == "mps":
        return DeviceInfo("metal", "Metal GPU", processor_name())
    if device.type == "cpu":
        threads = torch.get_num_threads()
        return DeviceInfo(
            "cpu", "CPU",
            f"{processor_name()} · {threads} thread{'s' * (threads != 1)}")
    return DeviceInfo("gpu", device.type.upper(), str(device))


def device_badges_html(models: DemoModels) -> str:
    """One badge naming the device both models run on, or one per model."""
    devices = [("", describe_device(models.offline.device))]
    realtime = describe_device(models.realtime.device)
    if realtime != devices[0][1]:
        devices = [("Offline: ", devices[0][1]), ("Online: ", realtime)]
    badges = []
    for prefix, info in devices:
        badges.append(
            f'<span class="device-badge device-{html.escape(info.kind)}"'
            f' title="Inference device: {html.escape(info.name)},'
            f' {html.escape(info.detail)}">'
            '<span class="device-dot"></span>'
            f"<span>{html.escape(prefix)}Running on"
            f" <strong>{html.escape(info.name)}</strong></span>"
            f'<span class="device-detail">· {html.escape(info.detail)}</span>'
            "</span>")
    return f'<div class="device-badges">{"".join(badges)}</div>'


def header_html(models: DemoModels | None = None) -> str:
    """Returns the shared header: logo, title and where inference runs."""
    logo = ""
    if LOGO.exists():
        svg = LOGO.read_text(encoding="utf-8").replace(
            "<svg ", '<svg fill="currentColor" ', 1)
        logo = f'<div class="logo">{svg}</div>'
    badges = device_badges_html(models) if models is not None else ""
    return (f'<div class="demo-header">{logo}<h1>Voice Isolator</h1>'
            "<p>Your voice, with the background removed.</p>"
            f"{badges}</div>")


def mono_float(audio):
    """Converts browser PCM to finite mono without normalizing its level."""
    rate, x = audio
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        maximum = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maximum
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not np.isfinite(x).all() or x.size == 0:
        raise gr.Error("Please provide a non-empty audio recording.")
    return rate, x


def needs_soxr(rate: int) -> bool:
    """Whether the library's resampler would need a huge kernel for ``rate``.

    Its kernel is sized by the reduced ratio of the two rates: small for the
    usual rates, gigabytes for one that shares few factors with 48 kHz
    (44101 Hz, 7993 Hz).
    """
    return SAMPLE_RATE // math.gcd(int(rate), SAMPLE_RATE) > 1000


def _soxr_48k(x: np.ndarray, rate: int) -> np.ndarray:
    # pylint: disable-next=import-outside-toplevel
    import soxr  # in the demo extra, beside Gradio

    return soxr.resample(x, rate, SAMPLE_RATE).astype(np.float32)


def to_48k(x: np.ndarray, rate: int) -> np.ndarray:
    """Resamples mono float32 audio to 48 kHz as the library does.

    Rates for which ``needs_soxr`` go through soxr instead.
    """
    x = np.asarray(x, dtype=np.float32)
    if rate == SAMPLE_RATE:
        return x
    if needs_soxr(rate):
        return _soxr_48k(x, rate)
    return voice_isolation.resample(torch.from_numpy(x), rate,
                                    SAMPLE_RATE).numpy()


def spectrogram_png(wav, sr=SAMPLE_RATE):
    """Builds a bounded 0-24 kHz spectrogram image without plotting figures.

    Audio at another rate is resampled to 48 kHz first (soxr: fast enough
    for a preview of any length), so every panel shares one frequency axis.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if sr != SAMPLE_RATE:
        wav = _soxr_48k(wav, sr)
    size = 1024
    wav = np.pad(wav, (0, max(0, size - len(wav))))
    # At most 640 columns, regardless of the uploaded recording's duration.
    starts = np.linspace(0,
                         len(wav) - size,
                         min(640, max(1,
                                      len(wav) // 256)),
                         dtype=int)
    frames = wav[starts[:, None] + np.arange(size)] * np.hanning(size)
    amplitude = np.abs(np.fft.rfft(frames, axis=1)) / (size / 2)
    db = 20 * np.log10(np.maximum(amplitude, 1e-5))
    values = np.clip((db + 100) / 100, 0, 1)
    palette = np.array([[7, 13, 27], [44, 34, 99], [121, 53, 137],
                        [225, 98, 107], [255, 210, 125]])
    index = np.minimum((values * 4).astype(int), 3)
    fraction = (values * 4 - index)[..., None]
    rgb = palette[index] * (1 - fraction) + palette[index + 1] * fraction
    return np.flip(rgb.transpose(1, 0, 2), axis=0).astype(np.uint8)


def pcm16(wav) -> np.ndarray:
    """Returns a preview at the original level, clipping above full scale."""
    return np.round(np.clip(wav, -1, 1) * 32767).astype(np.int16)


def load_input(audio) -> tuple[int, np.ndarray]:
    """Decodes the Offline input to mono float32 at its own rate and level.

    ``audio`` is a file path, as the input delivers uploads, samples and
    recordings, or a (rate, samples) pair. libsndfile reads WAV (float and
    extensible headers included), FLAC, OGG and MP3 without converting them;
    other formats go through Gradio's decoder. Inputs longer than
    ``MAX_OFFLINE_SECONDS`` are refused before decoding where the header
    says how long they are.
    """
    if audio is None:
        raise gr.Error(
            "Upload a file, pick a sample or record some audio first.")
    if isinstance(audio, (str, os.PathLike)):
        try:
            info = sf.info(audio)
            _check_duration(info.frames / info.samplerate)
            data, rate = sf.read(audio, dtype="float32", always_2d=True)
        except sf.LibsndfileError:
            try:
                rate, data = gr.processing_utils.audio_from_file(str(audio))
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Gradio's decoder raises whatever its backends raise.
                raise gr.Error(
                    "This audio format could not be read.") from error
        audio = (rate, data)
    rate, x = mono_float(audio)
    _check_duration(x.size / rate)
    return rate, x


def _check_duration(seconds: float) -> None:
    if seconds > MAX_OFFLINE_SECONDS:
        raise gr.Error(
            f"The Offline tab takes up to {MAX_OFFLINE_SECONDS // 60}"
            " minutes of audio; this input is longer.")


class LatestRequests:
    """The latest Offline request of each browser.

    Gradio's cancellation cannot stop a request whose thread already runs:
    an isolation superseded by a newer one from the same browser should give
    way when it gets the model, instead of running anyway.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = {}

    def begin(self, request: gr.Request | None) -> abc.Callable[[], bool]:
        """Records a new request; returns whether it has been superseded."""
        browser = getattr(request, "session_hash", None)
        with self._lock:
            token = self._latest.get(browser, 0) + 1
            self._latest[browser] = token
        return lambda: self._latest.get(browser) != token

    def forget(self, request: gr.Request | None = None) -> None:
        """Drops a closed browser's entry."""
        with self._lock:
            self._latest.pop(getattr(request, "session_hash", None), None)


def run_offline(audio,
                models: DemoModels,
                superseded: abc.Callable[[], bool] | None = None):
    """Isolates a recording once and derives its aligned background residual.

    A request ``superseded`` by the time the model is free returns no change.
    """
    rate, x = load_input(audio)
    if rate != SAMPLE_RATE and needs_soxr(rate):
        x, rate = to_48k(x, rate), SAMPLE_RATE
    start = time.perf_counter()
    with models.offline_lock:
        if superseded is not None and superseded():
            return (gr.skip(),) * 5
        voice = models.offline.isolate(x, rate, output_sample_rate=48000)
    aligned = to_48k(x, rate)
    if voice.shape != aligned.shape:
        raise gr.Error(
            "The isolated voice and input must have the same length.")
    background = aligned - voice
    seconds = len(voice) / 48000
    elapsed = time.perf_counter() - start
    device = describe_device(models.offline.device).name
    return ((48000, pcm16(voice)), spectrogram_png(voice, 48000),
            (48000, pcm16(background)), spectrogram_png(background, 48000),
            f"{seconds:.1f} s processed in {elapsed:.2f} s · "
            f"{seconds / max(elapsed, 1e-6):.1f}× real time on {device}")


def input_changed(audio):
    """Clears the previous outputs when the input changes.

    An input that cannot be isolated still clears them, with a warning.
    """
    image = None
    if audio is not None:
        try:
            rate, x = load_input(audio)
            image = spectrogram_png(x, rate)
        except gr.Error as error:
            gr.Warning(error.message)
    return image, None, None, None, None, ""


def _auth(spec):
    """'user:password' -> (user, password); None when empty."""
    if not spec:
        return None
    user, sep, password = spec.partition(":")
    if not sep or not user or not password:
        raise SystemExit(
            "--auth / VOICE_ISOLATION_DEMO_AUTH must be 'user:password'.")
    return (user, password)


def build_demo(models: DemoModels,
               examples: example_audio.Examples | None = None) -> gr.Blocks:
    """Builds offline file isolation and a stateful live microphone tab."""
    realtime_device = describe_device(models.realtime.device)
    controller = online.Online(models.realtime, models.realtime_lock,
                               realtime_device.name)
    latest = LatestRequests()

    def isolate_offline(audio, request: gr.Request):
        return run_offline(audio, models, superseded=latest.begin(request))

    with gr.Blocks(title="Inworld Voice Isolator",
                   analytics_enabled=False,
                   delete_cache=(600, 3600)) as demo:
        gr.HTML(header_html(models))
        with gr.Tabs(elem_id="studio-tabs"):
            with gr.Tab("Offline", id="offline"):
                gr.Markdown(
                    "Upload or record audio, then isolate the voice and"
                    f" background. Model: `{models.offline_name}`, the"
                    " offline model. It reads each window of audio whole,"
                    " before and after every moment, for the best quality."
                    " The Online tab runs a different model.")
                # The file as uploaded, unconverted: load_input decodes it.
                inp = gr.Audio(sources=["upload", "microphone"],
                               type="filepath",
                               format=None,
                               label="Input",
                               render=False,
                               elem_id="offline-input")
                if examples and examples.rows:
                    with gr.Column(variant="panel"):
                        gr.Markdown("### Samples")
                        picker = gr.Examples(examples=examples.rows,
                                             inputs=inp,
                                             label=None,
                                             example_labels=examples.labels,
                                             examples_per_page=max(
                                                 10, len(examples.rows)))
                        picker.dataset.show_label = False
                with gr.Row(elem_id="offline-grid"):
                    with gr.Column():
                        inp.render()
                        spec_in = gr.Image(label="Input spectrogram",
                                           show_label=False,
                                           height=SPEC_HEIGHT,
                                           buttons=[],
                                           interactive=False,
                                           elem_classes=["spectrogram"])
                    with gr.Column():
                        out = gr.Audio(label="Isolated voice",
                                       interactive=False,
                                       format="wav",
                                       buttons=["download"],
                                       elem_id="offline-voice")
                        spec_out = gr.Image(label="Voice spectrogram",
                                            show_label=False,
                                            height=SPEC_HEIGHT,
                                            buttons=[],
                                            interactive=False,
                                            elem_classes=["spectrogram"])
                    with gr.Column():
                        background = gr.Audio(label="Background",
                                              interactive=False,
                                              format="wav",
                                              buttons=["download"],
                                              elem_id="offline-background")
                        spec_background = gr.Image(
                            label="Background spectrogram",
                            show_label=False,
                            height=SPEC_HEIGHT,
                            buttons=[],
                            interactive=False,
                            elem_classes=["spectrogram"])
                btn = gr.Button("Isolate voice & background",
                                variant="primary",
                                elem_id="offline-run")
                info = gr.Markdown("")
                outputs = [out, spec_out, background, spec_background, info]
                isolation = btn.click(isolate_offline,
                                      inputs=inp,
                                      outputs=outputs,
                                      concurrency_id="offline-model",
                                      concurrency_limit=1,
                                      api_name="isolate_offline")
                # A new input cancels this browser's isolation in flight, so
                # its stale result cannot land after the outputs were
                # cleared; the preview itself never waits for the model.
                inp.change(input_changed,
                           inputs=inp,
                           outputs=[spec_in, *outputs],
                           cancels=[isolation],
                           concurrency_id="offline-preview",
                           concurrency_limit=4,
                           api_name=False)
            with gr.Tab("Online", id="online"):
                lookahead_ms = round(models.realtime.lookahead_frames *
                                     models.realtime.frame_seconds * 1000)
                intro = (
                    "Watch the clean voice update as you speak. Model:"
                    f" `{models.realtime_name}`, a separate model trained for"
                    " live audio. It hears what came before and"
                    f" {lookahead_ms} ms ahead, so on the same audio its"
                    " result differs from the Offline tab's. Pause to listen"
                    " without echo. Each take lasts up to 60 seconds.")
                if realtime_device.kind == "cpu":
                    intro += (" This demo runs on the CPU, which is usually"
                              " slower than live audio: capture then pauses"
                              " early and the take finishes.")
                gr.Markdown(intro)
                state = gr.State(None,
                                 time_to_live=120,
                                 delete_callback=online.close_recording)
                studio = gr.HTML(
                    value={},
                    html_template=(ASSETS /
                                   "online.html").read_text(encoding="utf-8"),
                    css_template=(ASSETS /
                                  "online.css").read_text(encoding="utf-8"),
                    js_on_load=(ASSETS /
                                "online.js").read_text(encoding="utf-8"),
                    elem_id="live-studio")
                # Below the panel's two cards, in the same order: what the
                # microphone heard, then the voice isolated from it.
                with gr.Row(elem_id="online-results", equal_height=True):
                    original = gr.Audio(label="Microphone input",
                                        interactive=False,
                                        format="wav",
                                        buttons=["download"],
                                        elem_id="online-input")
                    player = gr.Audio(label="Isolated voice",
                                      interactive=False,
                                      format="wav",
                                      buttons=["download"],
                                      elem_id="online-result")
                # Outside the queue, so a live take never waits behind
                # Offline jobs; the model's lock alone orders device work.
                studio.record_start(controller.start,
                                    inputs=state,
                                    outputs=[state, studio, player, original],
                                    api_name=False,
                                    queue=False,
                                    show_progress="hidden")
                studio.audio_chunk(controller.push,
                                   trigger_mode="multiple",
                                   inputs=state,
                                   outputs=[state, studio],
                                   api_name=False,
                                   queue=False,
                                   show_progress="hidden")
                studio.record_finish(controller.finish,
                                     inputs=state,
                                     outputs=[state, studio, player, original],
                                     api_name=False,
                                     queue=False,
                                     show_progress="hidden")
        demo.unload(controller.release)
        demo.unload(latest.forget)
        # Gradio copies each sample into its cache once, here, and keeps
        # serving that copy; the hourly cache sweep must not delete it.
        inp.keep_in_cache.update(set().union(*demo.temp_file_sets))
    return demo


def main():
    """Loads both models before accepting any offline or online requests."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline-model",
                    default=os.environ.get("VOICE_ISOLATION_OFFLINE_MODEL",
                                           registry.REPO_ID))
    ap.add_argument("--online-model",
                    default=os.environ.get("VOICE_ISOLATION_ONLINE_MODEL",
                                           registry.REPO_ID))
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision",
                    default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--url-file", default="")
    ap.add_argument("--examples",
                    default=os.environ.get(
                        "VOICE_ISOLATION_EXAMPLES",
                        str(example_audio.DEFAULT_DIRECTORY)))
    ap.add_argument("--auth",
                    default=os.environ.get("VOICE_ISOLATION_DEMO_AUTH", ""))
    args = ap.parse_args()
    auth = _auth(args.auth)
    examples = None
    if args.examples:
        directory = pathlib.Path(args.examples).resolve()
        caches = (gr.utils.get_upload_folder(), gr.utils.get_cache_folder())
        if any(
                directory.is_relative_to(pathlib.Path(cache).resolve())
                for cache in caches):
            ap.error("Example audio must be outside Gradio's upload "
                     "and examples cache directories.")
        try:
            examples = example_audio.load_examples(directory)
        except ValueError as error:
            ap.error(str(error))
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    print("Loading offline and realtime models…", flush=True)
    models = load_models(args.offline_model, args.online_model, args.device,
                         args.precision)
    warm_up(models)
    offline_device = describe_device(models.offline.device)
    realtime_device = describe_device(models.realtime.device)
    print(
        f"Offline ready on {offline_device.name} ({offline_device.detail}); "
        f"online ready on {realtime_device.name} ({realtime_device.detail}).",
        flush=True)
    demo = build_demo(models, examples)
    _, local_url, share_url = demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        auth=auth,
        prevent_thread_lock=True,
        allowed_paths=[row[0] for row in examples.rows] if examples else [],
        footer_links=["gradio", "settings"],
        max_file_size=MAX_UPLOAD_SIZE,
        css=CSS,
        # The Soft theme's colours with the default theme's typeface.
        theme=gr.themes.Soft(primary_hue="emerald",
                             neutral_hue="slate",
                             font=(gr.themes.LocalFont("Source Sans Pro"),
                                   "ui-sans-serif", "system-ui", "sans-serif")))
    lines = [f"local:  {local_url}"
            ] + ([f"share:  {share_url}"] if share_url else [])
    print("\n".join(lines), flush=True)
    if args.url_file:
        pathlib.Path(args.url_file).write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    demo.block_thread()


if __name__ == "__main__":
    main()
