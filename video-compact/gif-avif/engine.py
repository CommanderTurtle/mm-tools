from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable


DITHERS = {
    "none": "0",
    "sierra2_4a": "sierra2_4a",
    "floyd_steinberg": "floyd_steinberg",
    "bayer": "bayer",
    "atkinson": "atkinson",
}


@dataclass(slots=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    bytes: int
    codec: str
    video_ordinal: int
    audio_codec: str | None


@dataclass(slots=True)
class AnimationOptions:
    format: str = "gif"
    start: float = 0.0
    end: float = 0.0
    width: int = 640
    fps: float = 15.0
    speed: float = 1.0
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    rotate: str = "none"
    flip: str = "none"
    loop: int = 0
    gif_colors: int = 256
    gif_dither: str = "sierra2_4a"
    gif_palette: str = "diff"
    gif_alpha_threshold: int = 128
    gif_engine: str = "cpu"
    avif_quality: int = 78
    avif_effort: int = 6
    avif_engine: str = "software"
    avif_10bit: bool = False
    webm_quality: int = 78
    webm_engine: str = "nvenc"
    webm_audio: bool = True

    def validate(self) -> None:
        if self.format not in {"gif", "avif", "webm"}:
            raise ValueError("format must be gif, avif, or webm")
        if self.start < 0 or self.end < 0 or (self.end and self.end <= self.start):
            raise ValueError("end must be greater than start")
        if self.width and not 16 <= self.width <= 16384:
            raise ValueError("width must be zero or between 16 and 16384")
        if not 1 <= self.fps <= 120:
            raise ValueError("fps must be between 1 and 120")
        if not 0.1 <= self.speed <= 8:
            raise ValueError("speed must be between 0.1x and 8x")
        crop_values = (self.crop_x, self.crop_y, self.crop_width, self.crop_height)
        if any(value < 0 for value in crop_values):
            raise ValueError("crop values cannot be negative")
        if bool(self.crop_width) != bool(self.crop_height):
            raise ValueError("crop width and height must both be set")
        if self.rotate not in {"none", "cw", "ccw", "180"}:
            raise ValueError("invalid rotation")
        if self.flip not in {"none", "horizontal", "vertical", "both"}:
            raise ValueError("invalid flip")
        if not 0 <= self.loop <= 65535:
            raise ValueError("loop must be between 0 and 65535")
        if not 2 <= self.gif_colors <= 256:
            raise ValueError("GIF colors must be between 2 and 256")
        if self.gif_dither not in DITHERS:
            raise ValueError("unsupported GIF dither")
        if self.gif_palette not in {"full", "diff", "single"}:
            raise ValueError("GIF palette mode must be full, diff, or single")
        if not 0 <= self.gif_alpha_threshold <= 255:
            raise ValueError("alpha threshold must be between 0 and 255")
        if self.gif_engine not in {"cpu", "cuda"}:
            raise ValueError("GIF engine must be cpu or cuda")
        if not 0 <= self.avif_quality <= 100 or not 0 <= self.avif_effort <= 8:
            raise ValueError("AVIF quality must be 0-100 and effort 0-8")
        if self.avif_engine not in {"software", "nvenc"}:
            raise ValueError("AVIF engine must be software or nvenc")
        if not 0 <= self.webm_quality <= 100:
            raise ValueError("WebM quality must be between 0 and 100")
        if self.webm_engine not in {"software", "nvenc"}:
            raise ValueError("WebM engine must be software or nvenc")


def _rate(value: str) -> float:
    numerator, _, denominator = value.partition("/")
    try:
        return float(numerator) / max(float(denominator or "1"), 1e-9)
    except ValueError:
        return 0.0


def probe(path: Path) -> MediaInfo:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    all_streams = payload.get("streams", [])
    streams = [stream for stream in all_streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in all_streams if stream.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("input has no video stream")
    # Animated AVIF contains a one-frame primary image followed by its animation
    # track. Prefer the stream with the most frames/duration, while ordinary
    # videos naturally retain ordinal zero.
    ordinal, video = max(
        enumerate(streams),
        key=lambda item: (
            int(item[1].get("nb_frames") or 0),
            float(item[1].get("duration") or 0),
        ),
    )
    return MediaInfo(
        duration=float(video.get("duration") or payload.get("format", {}).get("duration") or 0),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_rate(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")),
        bytes=path.stat().st_size,
        codec=str(video.get("codec_name") or "unknown"),
        video_ordinal=ordinal,
        audio_codec=str(audio_streams[0].get("codec_name") or "unknown") if audio_streams else None,
    )


def available_encoders() -> set[str]:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    return {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2 and line.lstrip().startswith("V")}


def _filters(
    options: AnimationOptions,
    *,
    even_dimensions: bool = False,
    include_scale: bool = True,
) -> list[str]:
    filters: list[str] = []
    if options.crop_width and options.crop_height:
        filters.append(f"crop={options.crop_width}:{options.crop_height}:{options.crop_x}:{options.crop_y}")
    if options.rotate == "cw":
        filters.append("transpose=clock")
    elif options.rotate == "ccw":
        filters.append("transpose=cclock")
    elif options.rotate == "180":
        filters.extend(["hflip", "vflip"])
    if options.flip in {"horizontal", "both"}:
        filters.append("hflip")
    if options.flip in {"vertical", "both"}:
        filters.append("vflip")
    if include_scale and options.width:
        width = options.width - (options.width % 2 if options.format == "avif" or even_dimensions else 0)
        filters.append(f"scale={width}:-2:flags=lanczos")
    elif include_scale and (options.format == "avif" or even_dimensions):
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos")
    filters.append(f"fps={options.fps:g}")
    if abs(options.speed - 1.0) > 0.0001:
        filters.append(f"setpts=PTS/{options.speed:g}")
    return filters


def _gif_cuda_filters(options: AnimationOptions) -> list[str]:
    """Use CUDA for decode/color conversion and resize when geometry permits."""
    has_cpu_geometry = bool(options.crop_width) or options.rotate != "none" or options.flip != "none"
    if has_cpu_geometry:
        return [
            "scale_cuda=iw:ih:format=yuv420p:passthrough=0",
            "hwdownload",
            "format=yuv420p",
            *_filters(options),
        ]

    width = str(options.width) if options.width else "iw"
    height = "-2" if options.width else "ih"
    return [
        f"scale_cuda={width}:{height}:interp_algo=lanczos:format=yuv420p:passthrough=0",
        "hwdownload",
        "format=yuv420p",
        *_filters(options, include_scale=False),
    ]


def _atempo_filters(speed: float) -> list[str]:
    """Split arbitrary supported speeds into FFmpeg atempo's 0.5-2.0 range."""
    remaining = speed
    factors: list[float] = []
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.0001:
        factors.append(remaining)
    return [f"atempo={factor:g}" for factor in factors]


def build_command(input_path: Path, output_path: Path, options: AnimationOptions) -> tuple[list[str], float]:
    options.validate()
    info = probe(input_path)
    if options.crop_width and (
        options.crop_x + options.crop_width > info.width
        or options.crop_y + options.crop_height > info.height
    ):
        raise ValueError("crop rectangle extends outside the source video")
    duration = max(0.0, (options.end or info.duration) - options.start) / options.speed
    command = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "warning"]
    if options.start:
        command.extend(["-ss", f"{options.start:.6f}"])
    if options.format == "gif" and options.gif_engine == "cuda":
        if "cuda" not in subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, check=True
        ).stdout.split():
            raise ValueError("CUDA hardware acceleration is unavailable in this FFmpeg build")
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(input_path)])
    if options.end:
        command.extend(["-t", f"{(options.end - options.start):.6f}"])
    chain = ",".join(
        _gif_cuda_filters(options)
        if options.format == "gif" and options.gif_engine == "cuda"
        else _filters(options, even_dimensions=options.format == "webm")
    )

    if options.format == "gif":
        palette = f"palettegen=max_colors={options.gif_colors}:reserve_transparent=1:stats_mode={options.gif_palette}"
        use = f"paletteuse=dither={DITHERS[options.gif_dither]}:alpha_threshold={options.gif_alpha_threshold}"
        if options.gif_palette == "diff":
            use += ":diff_mode=rectangle"
        if options.gif_palette == "single":
            use += ":new=1"
        command.extend(["-filter_complex", f"[0:v:{info.video_ordinal}]{chain},split[v0][v1];[v0]{palette}[p];[v1][p]{use}[out]"])
        command.extend(["-map", "[out]", "-an", "-loop", str(options.loop), "-gifflags", "+transdiff"])
    elif options.format == "avif":
        command.extend(["-map", f"0:v:{info.video_ordinal}", "-vf", chain, "-an"])
        if options.avif_engine == "nvenc":
            if "av1_nvenc" not in available_encoders():
                raise ValueError("av1_nvenc is unavailable in this FFmpeg build")
            cq = max(0, min(51, round(51 - options.avif_quality * 0.43)))
            command.extend(["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"])
        else:
            if "libaom-av1" not in available_encoders():
                raise ValueError("libaom-av1 is unavailable in this FFmpeg build")
            crf = max(0, min(63, round(63 - options.avif_quality * 0.55)))
            command.extend(["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0", "-cpu-used", str(options.avif_effort), "-row-mt", "1", "-still-picture", "0"])
        command.extend(["-pix_fmt", "yuv420p10le" if options.avif_10bit else "yuv420p", "-loop", str(options.loop), "-f", "avif"])
    else:
        command.extend(["-map", f"0:v:{info.video_ordinal}", "-vf", chain])
        if options.webm_engine == "nvenc":
            if "av1_nvenc" not in available_encoders():
                raise ValueError("av1_nvenc is unavailable in this FFmpeg build")
            cq = max(0, min(51, round(51 - options.webm_quality * 0.43)))
            command.extend([
                "-c:v", "av1_nvenc", "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
            ])
        else:
            if "libvpx-vp9" not in available_encoders():
                raise ValueError("libvpx-vp9 is unavailable in this FFmpeg build")
            crf = max(4, min(63, round(63 - options.webm_quality * 0.55)))
            command.extend([
                "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
                "-deadline", "good", "-cpu-used", "2", "-row-mt", "1",
            ])
        command.extend(["-pix_fmt", "yuv420p"])
        if options.webm_audio:
            if not info.audio_codec:
                raise ValueError("The source has no audio stream to keep in the WebM output.")
            command.extend(["-map", "0:a:0", "-c:a", "libopus", "-b:a", "128k"])
            audio_filters = _atempo_filters(options.speed)
            if audio_filters:
                command.extend(["-af", ",".join(audio_filters)])
            command.append("-shortest")
        else:
            command.append("-an")
        command.extend(["-f", "webm"])
    command.extend(["-progress", "pipe:1", "-nostats", str(output_path)])
    return command, duration or info.duration


def _run_conversion(
    command: list[str],
    output_path: Path,
    duration: float,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_process: Callable[[subprocess.Popen[str]], None] | None = None,
    cancel: Event | None = None,
) -> MediaInfo:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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


def convert(
    input_path: Path,
    output_path: Path,
    options: AnimationOptions,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_process: Callable[[subprocess.Popen[str]], None] | None = None,
    cancel: Event | None = None,
) -> MediaInfo:
    command, duration = build_command(input_path, output_path, options)
    return _run_conversion(
        command, output_path, duration,
        on_progress=on_progress, on_process=on_process, cancel=cancel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local video-to-GIF/animated-AVIF/WebM converter")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("gif", "avif", "webm"), default="gif")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--colors", type=int, default=256)
    parser.add_argument("--dither", choices=DITHERS, default="sierra2_4a")
    parser.add_argument("--palette", choices=("full", "diff", "single"), default="diff")
    parser.add_argument("--gif-engine", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quality", type=int, default=78)
    parser.add_argument("--effort", type=int, default=6)
    parser.add_argument("--avif-engine", choices=("software", "nvenc"), default="software")
    parser.add_argument("--ten-bit", action="store_true")
    parser.add_argument("--webm-engine", choices=("software", "nvenc"), default="nvenc")
    parser.add_argument("--webm-quality", type=int, default=78)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe must be on PATH")
    options = AnimationOptions(
        format=args.format, start=args.start, end=args.end, width=args.width,
        fps=args.fps, speed=args.speed, loop=args.loop, gif_colors=args.colors,
        gif_dither=args.dither, gif_palette=args.palette, avif_quality=args.quality,
        gif_engine=args.gif_engine, avif_effort=args.effort,
        avif_engine=args.avif_engine, avif_10bit=args.ten_bit,
        webm_engine=args.webm_engine, webm_quality=args.webm_quality,
        webm_audio=not args.no_audio,
    )
    before = probe(args.input)
    after = convert(args.input, args.output, options, on_progress=lambda _p, line: print(line))
    print(json.dumps({"before": asdict(before), "after": asdict(after)}, indent=2))


if __name__ == "__main__":
    main()
