# studio / Unlimited OCR 2

Ordered PDF and image pages, native BF16 recognition, layout regions, cross-page
text/table merges, equations and figure crops. Read the document beside its
source, inspect raw blocks, or export Markdown, layout JSON and a complete bundle.

```bash
uv run models/download_models.py unlimited-ocr2
cd visual-modeling/OCR
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8287`. Set `OCR_PORT` to use another port. Setup needs `uv`
and Bun; the native model uses CUDA. Model files remain in the shared store,
linked relative to this folder regardless of checkout name or depth.

Open PDFs or page images, arrange the reading order and select **Parse document**.
Auto tiles single-page inputs; Base uses a global view for each document page;
Gundam enables native dynamic tiling on every page. PDF DPI applies on upload.
Context and per-page token budgets remain explicit. Every uploaded page is kept;
partial native output is saved and labeled when a budget is reached.

This is the official Space's Transformers/DASH inference path with the version-2
document postprocessor. The separate SGLang unlimited-horizon backend has its own
runtime contract.

Documents and each parse are stored under `.work/documents`. **Documents** reopens
results or deletes a selected run. Parsing loads a worker on demand; stopping or
closing the server releases it. Completed and partial files survive interruption.
The example report is generated locally. Browser libraries are served locally.
