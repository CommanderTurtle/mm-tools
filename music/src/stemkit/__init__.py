"""StemKit vendor package (danielravina/stemkit, MIT, see README.md).

Runtime surface used by the mm-tools studios: ``stemkit.studio_api`` drives
the vendored ``python/separate.py`` (demucs) and ``python/roformer.py``
(Mel-Band Roformer vocals) as subprocesses with the JSON progress protocol
they already emit, and derives vocal-aligned lyric timestamps for the
studio visualizers.
"""

__version__ = "0.1.22"
