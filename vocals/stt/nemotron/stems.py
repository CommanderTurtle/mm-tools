import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from state import MODELS, PIN


def main():
    job = Path(sys.argv[1])
    root = MODELS / "stems"
    checkpoint = root / "955717e8-8726e21a.th"
    with checkpoint.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != PIN["stem_sha256"]:
            raise ValueError("Complete the nemotron model download: htdemucs checksum mismatch")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(job / "upload"),
                    "-map", "0:a:0", "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                    str(job / "original.wav")], check=True)
    # Demucs' explicit local repository prevents its default remote-weight lookup.
    subprocess.run([sys.executable, "-m", "demucs", "--repo", str(root), "-n", "955717e8",
                    "--two-stems", "vocals", "-d", "cpu", "--float32", "-j", "2", "-o",
                    str(job / "separated"), str(job / "original.wav")], check=True)
    output = job / "separated/955717e8/original"
    shutil.move(str(output / "vocals.wav"), job / "vocals.wav")
    shutil.move(str(output / "no_vocals.wav"), job / "music.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(job / "vocals.wav"),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(job / "audio.wav")], check=True)


if __name__ == "__main__":
    main()
