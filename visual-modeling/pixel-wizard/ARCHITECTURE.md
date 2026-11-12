# Pixel Wizard

Local WeVisOne-VL-4B image perception studio. Select `pixel-wizard` in
`models/download_models.py`, then:

```bash
cd visual-modeling/pixel-wizard
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8288`. `PIXEL_PORT` changes the port. Setup uses Python 3.12,
the native source and a relative link to the shared model store.

Upload, paste, capture or choose an example image. Select grounding, dense
detection, box/point counting, pointing, OCR, grounded OCR, caption/VQA,
segmentation, matting or similar-object detection. Draw reference boxes directly
on the input for similar objects or single-box matting. The latter supplies the
box to the native pixel decoder; its receipt identifies that box as user input.

Generation controls and an optional exact prompt override follow the released
engine. The model runs on CUDA in BF16, as in the reference Space. Each run loads
one worker and releases it afterward. Stop releases the worker; saved inputs and
completed results remain in `.work/runs` across restarts. Saved runs can be opened,
rerun or explicitly deleted.

Inspect each instance, category-colored boxes/points and native masks. Overlay,
green-screen, alpha and cutout views share the same output; display controls never
rerun inference. Segmentation uses the selected display threshold, while matting
retains continuous alpha. Replay reveals the saved response and geometry.

Export view saves a full-resolution PNG with the current display settings. The
bundle retains the source image, native overlay, response, normalized decoder
boxes, pixel-space annotations, individual masks, alpha, cutout and generation
settings. Native exports retain their original visualization independently of
the current display settings.

Weights and their conditions are listed in `models/which-ones.txt`.
