from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable


QUALITY_CRF = {
    "best": {"h264": 18, "h265": 20, "av1": 24, "vp9": 24},
    "balanced": {"h264": 23, "h265": 25, "av1": 30, "vp9": 31},
    "small": {"h264": 28, "h265": 29, "av1": 37, "vp9": 38},
}
SOFTWARE_ENCODERS = {"h264": "libx264", "h265": "libx265", "av1": "libsvtav1", "vp9": "libvpx-vp9"}
NVENC_ENCODERS = {"h264": "h264_nvenc", "h265": "hevc_nvenc", "av1": "av1_nvenc"}


@dataclass(slots=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    bytes: int
    video_codec: str
    audio_codec: str | None


@dataclass(slots=True)
class CompactOptions:
    codec: str = "h265"
    engine: str = "auto"
    quality: str = "balanced"
    width: int = 0
    remove_audio: bool = False
    audio_bitrate: int = 128
    start: float = 0.0
    end: float = 0.0
    fps: float = 0.0
    strip_metadata: bool = True
    faststart: bool = True

    def validate(self) -> None:
        if self.codec not in SOFTWARE_ENCODERS:
            raise ValueError("codec must be h264, h265, av1, or vp9")
        if self.engine not in {"auto", "software", "nvenc"}:
            raise ValueError("engine must be auto, software, or nvenc")
        if self.engine == "nvenc" and self.codec == "vp9":
            raise ValueError("VP9 has no NVENC encoder")
        if self.quality not in QUALITY_CRF:
            raise ValueError("quality must be best, balanced, or small")
        if self.width and not 64 <= self.width <= 16384:
            raise ValueError("width must be zero or between 64 and 16384")
        if self.audio_bitrate not in {64, 96, 128, 160, 192, 256, 320}:
            raise ValueError("unsupported audio bitrate")
        if self.start < 0 or self.end < 0 or (self.end and self.end <= self.start):
            raise ValueError("end must be greater than start")
        if self.fps and not 1 <= self.fps <= 240:
            raise ValueError("fps must be zero or between 1 and 240")


def _rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, _, denominator = value.partition("/")
    try:
        return float(numerator) / max(float(denominator or "1"), 1e-9)
    except ValueError:
        return 0.0


def probe(path: Path) -> MediaInfo:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("input has no video stream")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0)
    return MediaInfo(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_rate(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")),
        bytes=path.stat().st_size,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name")) if audio else None,
    )


def available_encoders() -> set[str]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    )
    return {
        line.split()[1]
        for line in result.stdout.splitlines()
        if len(line.split()) >= 2 and line.lstrip().startswith("V")
    }


def _encoder(options: CompactOptions, encoders: set[str]) -> tuple[str, bool]:
    hardware = NVENC_ENCODERS.get(options.codec)
    if options.engine == "nvenc" or (options.engine == "auto" and hardware in encoders):
        if not hardware or hardware not in encoders:
            raise ValueError(f"NVENC is not available for {options.codec}")
        return hardware, True
    software = SOFTWARE_ENCODERS[options.codec]
    if software not in encoders:
        raise ValueError(f"FFmpeg encoder is unavailable: {software}")
    return software, False


def _target_dimensions(info: MediaInfo, requested_width: int) -> tuple[int, int] | None:
    if not requested_width or not info.width or requested_width >= info.width:
        return None
    width = requested_width - requested_width % 2
    height = max(2, round(info.height * width / info.width))
    height -= height % 2
    return width, height


def build_command(input_path: Path, output_path: Path, options: CompactOptions) -> tuple[list[str], float]:
    options.validate()
    info = probe(input_path)
    encoder, hardware = _encoder(options, available_encoders())
    duration = max(0.0, (options.end or info.duration) - options.start)
    command = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "warning"]
    if options.start:
        command.extend(["-ss", f"{options.start:.6f}"])
    command.extend(["-i", str(input_path)])
    if options.end:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend(["-map", "0:v:0"])
    filters: list[str] = []
    dimensions = _target_dimensions(info, options.width)
    if dimensions:
        filters.append(f"scale={dimensions[0]}:{dimensions[1]}:flags=lanczos")
    if options.fps and (not info.fps or options.fps < info.fps - 0.01):
        filters.append(f"fps={options.fps:g}")
    if filters:
        command.extend(["-vf", ",".join(filters)])

    crf = QUALITY_CRF[options.quality][options.codec]
    command.extend(["-c:v", encoder])
    if hardware:
        preset = {"best": "p7", "balanced": "p6", "small": "p5"}[options.quality]
        command.extend(["-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"])
    elif options.codec in {"h264", "h265"}:
        preset = {"best": "slow", "balanced": "medium", "small": "slow"}[options.quality]
        command.extend(["-preset", preset, "-crf", str(crf)])
    elif options.codec == "av1":
        preset = {"best": "4", "balanced": "6", "small": "7"}[options.quality]
        command.extend(["-preset", preset, "-crf", str(crf), "-svtav1-params", "tune=0"])
    else:
        command.extend(["-crf", str(crf), "-b:v", "0", "-cpu-used", "2", "-row-mt", "1"])

    if options.remove_audio or not info.audio_codec:
        command.append("-an")
    else:
        command.extend(["-map", "0:a:0?", "-c:a", "libopus" if options.codec == "vp9" else "aac"])
        command.extend(["-b:a", f"{options.audio_bitrate}k"])
    if options.strip_metadata:
        command.extend(["-map_metadata", "-1", "-map_chapters", "-1"])
    if options.faststart and output_path.suffix.lower() == ".mp4":
        command.extend(["-movflags", "+faststart"])
    command.extend(["-progress", "pipe:1", "-nostats", str(output_path)])
    return command, duration or info.duration


def transcode(
    input_path: Path,
    output_path: Path,
    options: CompactOptions,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_process: Callable[[subprocess.Popen[str]], None] | None = None,
    cancel: Event | None = None,
) -> MediaInfo:
    command, duration = build_command(input_path, output_path, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if on_process:
        on_process(process)
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if cancel and cancel.is_set() and process.poll() is None:
            process.terminate()
        if line.startswith("out_time_us="):
            try:
                seconds = int(line.split("=", 1)[1]) / 1_000_000
                if on_progress:
                    on_progress(min(0.99, seconds / max(duration, 0.001)), line)
            except ValueError:
                pass
        elif line and on_progress:
            on_progress(-1, line)
    returncode = process.wait()
    if cancel and cancel.is_set():
        output_path.unlink(missing_ok=True)
        raise InterruptedError("conversion cancelled")
    if returncode:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg exited with status {returncode}")
    if on_progress:
        on_progress(1.0, "conversion complete")
    return probe(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local FFmpeg video compactor")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--codec", choices=SOFTWARE_ENCODERS, default="h265")
    parser.add_argument("--engine", choices=("auto", "software", "nvenc"), default="auto")
    parser.add_argument("--quality", choices=QUALITY_CRF, default="balanced")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--remove-audio", action="store_true")
    parser.add_argument("--audio-bitrate", type=int, default=128)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=0)
    parser.add_argument("--fps", type=float, default=0)
    parser.add_argument("--keep-metadata", action="store_true")
    parser.add_argument("--no-faststart", action="store_true")
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe must be on PATH")
    options = CompactOptions(
        codec=args.codec,
        engine=args.engine,
        quality=args.quality,
        width=args.width,
        remove_audio=args.remove_audio,
        audio_bitrate=args.audio_bitrate,
        start=args.start,
        end=args.end,
        fps=args.fps,
        strip_metadata=not args.keep_metadata,
        faststart=not args.no_faststart,
    )
    before = probe(args.input)
    after = transcode(args.input, args.output, options, on_progress=lambda p, line: print(line))
    print(json.dumps({"before": asdict(before), "after": asdict(after)}, indent=2))


if __name__ == "__main__":
    main()
