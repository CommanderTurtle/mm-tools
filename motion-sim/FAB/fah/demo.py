"""FAH demo (same as the `fah` command): image(s) or a video in -> overlay + JSON out.

    python demo.py --img hand.jpg --output_dir out/
    python demo.py --vid clip.mp4 --backbone s --output_dir out/
"""
from fah.cli import main

if __name__ == "__main__":
    main()
