# SPDX-License-Identifier: AGPL-3.0-only
import json
import os
import subprocess
import shutil
import time
import sys
import tempfile
from pathlib import Path
import gradio as gr
from state import LANE_STAMP

ROOT=Path(__file__).resolve().parent
MODELS={"FAB-S":("fab-s","auto"), "FAB-L":("fab-l","auto"),
        "FAB-H · mesh fit":("fab-h","mesh"), "FAB-H · dense-point fit":("fab-h","points")}
HAND_MESHES={"Automatic":"auto","MHR":"mhr","MANO":"mano","MHR + MANO":"both"}

def generate(video,model,mesh,robot,ego,hand_mesh,blur,ref,all_people,camera,stems):
    if not video: raise gr.Error("Choose a video.")
    if model not in MODELS or mesh not in ("SMPL","SOMA") or hand_mesh not in HAND_MESHES: raise gr.Error("Unknown mesh preset.")
    model,profile=MODELS[model]
    hand_mesh=HAND_MESHES[hand_mesh]
    pair=all((Path(os.environ["MANO_DIR"])/name).is_file() for name in ("MANO_RIGHT.pkl","MANO_LEFT.pkl"))
    if ego and hand_mesh in ("both","mano") and not pair:
        raise gr.Error("MANO needs both MANO_LEFT.pkl and MANO_RIGHT.pkl. Setup reads the pair from SOMA-X/assets/MANO or MANO_SOURCE.")
    if hand_mesh=="auto": hand_mesh="both" if pair else "mhr"
    if model=="fab-h" and robot: raise gr.Error("Use the G1 reference export with FAB-H; its checkpoint has no camera G1 head.")
    jobs=ROOT/".state"/"jobs"; jobs.mkdir(parents=True,exist_ok=True)
    for old in jobs.glob("fab-*"):
        if old.is_dir() and not old.is_symlink() and old.stat().st_mtime < time.time()-86400: shutil.rmtree(old)
    out=Path(tempfile.mkdtemp(prefix="fab-",dir=jobs))
    spec=out/"job.json"; spec.write_text(json.dumps(dict(video=video,out=str(out),model=model,profile=profile,mesh=mesh,robot=robot,ego=ego,hand_mesh=hand_mesh,blur=blur,ref=ref,all_people=all_people,camera=camera,stems=stems,source=LANE_STAMP)))
    with subprocess.Popen([sys.executable,str(ROOT.parents[1]/"models/studio-child.py"),sys.executable,str(ROOT/"worker.py"),str(spec)],stdin=subprocess.PIPE) as proc:
        if proc.wait():
            shutil.rmtree(out,ignore_errors=True)
            raise gr.Error("FAB could not finish. See the local terminal for the failed stage.")
    data=[out/"tracks.pkl",*sorted((out/"retarget").glob("*/person-*_g1.npz")),*sorted((out/"retarget").glob("*/person-*_g1.json")),*sorted((out/"layers").glob("*.mp4")),*sorted((out/"hands").glob("*.json"))]
    from gradio.utils import get_upload_folder
    exports=Path(tempfile.mkdtemp(prefix="fab-",dir=get_upload_folder()))
    try:
        for p in [*data,out/"atlas.mp4",out/"hands.mp4",out/"reference.mp4"]:
            if p.is_file(): shutil.move(str(p),exports/p.name)
        return str(exports/"atlas.mp4"),str(exports/"hands.mp4") if ego else None,str(exports/"reference.mp4") if (exports/"reference.mp4").is_file() else None,[str(exports/p.name) for p in data]
    finally: shutil.rmtree(out,ignore_errors=True)
with gr.Blocks(title="FAB",theme=gr.themes.Base(primary_hue="emerald",neutral_hue="slate"),delete_cache=(86400,86400)) as ui:
    gr.Markdown("# Motion atlas\nInput, fitted mesh, body parts and point contacts from a single take.")
    with gr.Row():
        with gr.Column(scale=1):
            video=gr.Video(label="Footage",sources=["upload"],include_audio=False)
            model=gr.Radio(list(MODELS),value="FAB-L",label="Body estimator")
            mesh=gr.Radio(["SMPL","SOMA"],value="SMPL",label="Fitted mesh")
            robot=gr.Checkbox(label="Show direct camera-space G1 head")
            ego=gr.Checkbox(label="Head-mounted footage · add FAH hand view")
            hand_mesh=gr.Radio(list(HAND_MESHES),value="Automatic",label="Hand mesh",visible=False)
            ego.change(lambda enabled: gr.update(visible=enabled),ego,hand_mesh)
            model.change(lambda name: gr.update(value=False,interactive=False) if MODELS[name][0]=="fab-h" else gr.update(interactive=True),model,robot)
            blur=gr.Checkbox(label="Blur detected frontal faces",value=True)
            ref=gr.Checkbox(label="Overlay and export FAB-Retarget’s 50 Hz G1 reference")
            all_people=gr.Checkbox(label="Retarget every person (default: longest continuous track)",visible=False)
            camera=gr.File(label="Camera trajectory · optional FAB world NPZ; omit for fixed-camera footage",file_types=[".npz"])
            ref.change(lambda enabled: gr.update(visible=enabled),ref,all_people)
            stems=gr.Checkbox(label="Export separate full-frame layers")
            go=gr.Button("Track motion",variant="primary")
        with gr.Column(scale=2):
            view=gr.Video(label="Motion atlas"); hands=gr.Video(label="Head-mounted hands"); motion=gr.Video(label="FAB-Retarget · 50 Hz kinematic reference")
            files=gr.File(label="Native tracks and references",file_count="multiple")
    ex=Path(os.environ.get("FAB_HOME") or ROOT.parents[1]/"models/fab")/"demo"
    examples=[[str(p)] for p in sorted(ex.glob("*.mp4"))]
    if examples: gr.Examples(examples,[video])
    go.click(generate,[video,model,mesh,robot,ego,hand_mesh,blur,ref,all_people,camera,stems],[view,hands,motion,files],concurrency_limit=1)
if __name__=="__main__": ui.queue().launch(server_name="127.0.0.1",allowed_paths=[str(ex)],blocked_paths=[str(ROOT/".state/jobs")],server_port=int(os.environ["GRADIO_SERVER_PORT"]))
