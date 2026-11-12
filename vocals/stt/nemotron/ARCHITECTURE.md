# Nemotron

Local transcription, speaker timelines, subtitles, karaoke and audio questions.

```bash
uv run models/download_models.py nemotron
cd vocals/stt/nemotron
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8289` (`NEMOTRON_PORT` overrides the port). Linux, a
Blackwell GPU, `uv`, FFmpeg and CUDA 13 `nvcc` on PATH are required. Setup
links this studio to the shared model store at any checkout name or depth.

Upload audio/video, record a microphone, or explicitly import a media/episode URL.
Setup includes yt-dlp, its matching EJS components and a local Deno runtime.
Choose a recipe, edit its instruction, and run. Click words or cues to seek;
filter speakers, search, follow playback, change speed, or show video subtitles.
Add per-cue language notes to filter and seek a language timeline; these saved,
downloadable annotations remain separate from the native model response.
Karaoke separates vocals on CPU before transcription and provides original,
music and vocal playback with an adjustable vocal mix. Exports include native
JSON, text, word timing, SRT, VTT and LRC. Timing exports appear when alignment
succeeds. Saved runs can be reprocessed; **Continue conversation** adds an audio
question to the selected run's saved conversation.

The checkpoint retains its native NVFP4/FP8 mixed weights, BF16 compute and FP8
KV. The advanced panel controls context, output budget, GPU memory allocation,
MTP8 and reasoning. Native ReplaySSM and the CTC middleware remain enabled.
The published 65,536-context reference profile uses a 96 GB GPU; full-hour audio
capacity on a smaller card depends on those explicit budgets. Long recordings
can need over 16,384 output tokens. Limit/alignment warnings preserve the response.

Models start for submitted jobs and exit afterward; Stop closes the owned process
tree. Inputs, settings, transcripts and logs live in `.work/runs`; interrupted
runs remain available. Remove a saved run to delete its local files. First use
compiles native kernels into `.work/kernels-vllm032`. Model loading is offline;
only an explicitly supplied media URL contacts a remote service. Model conditions are
listed in the downloader inventory.
