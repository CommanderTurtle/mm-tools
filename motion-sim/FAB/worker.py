# SPDX-License-Identifier: AGPL-3.0-only
import json
import pickle
import subprocess
import sys
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from layers import Layers
from track_refs import cameras, solve

ROOT=Path(__file__).resolve().parent
BIN=Path(sys.executable).parent
def run(*args, cwd=None):
    subprocess.run(list(map(str,args)),check=True,cwd=cwd)
def main(video,out,model,profile,mesh,robot,ego,hand_mesh,blur,ref,all_people,camera,stems):
    out=Path(out); video=str(Path(video).resolve())
    with_camera=[]
    if camera:
        header=cv2.VideoCapture(video)
        try: _,K=cameras(camera,int(header.get(cv2.CAP_PROP_FRAME_COUNT)),header.get(cv2.CAP_PROP_FPS) or 30)
        finally: header.release()
        if K is not None: with_camera=["--K",",".join(map(str,(K[0,0],K[1,1],K[0,2],K[1,2])))]
    # The pickle below is written by our own child, never deserialized from an upload.
    result=out/"tracks.pkl"
    run(BIN/"fab-video",video,"--model",model,"--profile",profile,*with_camera,"--no-render","--results",result,cwd=ROOT/"fab")
    with result.open("rb") as f: tracks=pickle.load(f)
    if not tracks["frames"]: raise ValueError("No decodable frames.")
    w,h=tracks["size"]; fps=tracks["fps"]
    poses,_=cameras(camera,len(tracks["frames"]),fps)
    replay=solve(tracks["frames"],fps,poses,out,all_people) if ref else {}
    layers=Layers(tracks["K"],w,h,mesh,robot or ref)
    cap=cv2.VideoCapture(video)
    face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml") if blur else None
    ff=imageio_ffmpeg.get_ffmpeg_exe(); sw=min(720,w)//2*2; sh=max(2,round(h*sw/w/2)*2)
    writer=imageio_ffmpeg.write_frames(str(out/"atlas.mp4"),(sw*3,sh*2),fps=fps,pix_fmt_in="bgr24",macro_block_size=1)
    writer.send(None)
    separate={}
    try:
        if stems:
            (out/"layers").mkdir()
            for name in ("input","g1","mesh","parts","contacts","dense"):
                if (name=="g1" and not (robot or ref)) or (model=="fab-h" and (name=="parts" or (name=="dense" and profile=="mesh"))): continue
                stream=imageio_ffmpeg.write_frames(str(out/"layers"/f"{name}.mp4"),(w-w%2,h-h%2),fps=fps,pix_fmt_in="bgr24",macro_block_size=1)
                stream.send(None); separate[name]=stream
        for idx,frame in enumerate(tracks["frames"]):
            ok,img=cap.read()
            if not ok: raise ValueError("Video ended before the recorded tracks.")
            if face is not None:
                for x,y,fw,fh in face.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.12,5):
                    roi=img[y:y+fh,x:x+fw]; roi[:]=cv2.GaussianBlur(roi,(0,0),max(fw,fh)/7)
            people=frame["people"]
            panels=layers.panels(img,people,robot and not ref,512,replay.get(idx,()),poses[idx])
            tiles=[]
            labels=("Input",("G1 / retargeted reference" if ref else "G1 / camera head") if (robot or ref) else "Input",mesh,
                    "Input / no part head" if model=="fab-h" else "Body parts",
                    "Keypoints" if model=="fab-h" else "Keypoints / static contact",
                    "Input / no dense head run" if profile=="mesh" else "Dense points / depth")
            for name,field,label in zip(("input","g1","mesh","parts","contacts","dense"),panels,labels,strict=True):
                if name in separate: separate[name].send(np.ascontiguousarray(field[:h-h%2,:w-w%2]))
                tile=cv2.resize(field,(sw,sh));cv2.rectangle(tile,(0,0),(min(sw,235),27),(29,28,27),-1)
                cv2.putText(tile,label.replace("·","/"),(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,240,235),1,cv2.LINE_AA);tiles.append(tile)
            sheet=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]); writer.send(sheet)
            if idx==0: cv2.imwrite(str(out/"poster.jpg"),sheet)
    finally:
        writer.close(); cap.release(); layers.renderer.delete()
        for stream in separate.values(): stream.close()
    if ego:
        # Detect faces on the rendered hand clip; never soften the model's input.
        run(ROOT/".hands/bin/fah","--vid",video,"--mode","video","--mesh",hand_mesh,"--dense","--parts","--contact","--output_dir",out/"hands",cwd=ROOT/"fah")
        clip=next((out/"hands").glob("*.mp4"))
        if not blur: run(ff,"-loglevel","error","-i",clip,"-an","-pix_fmt","yuv420p",out/"hands.mp4")
        else:
            c=cv2.VideoCapture(str(clip)); wh=tuple(int(c.get(i)) for i in (3,4))
            stream=imageio_ffmpeg.write_frames(str(out/"hands.mp4"),wh,fps=c.get(5) or fps,pix_fmt_in="bgr24")
            stream.send(None)
            try:
                while True:
                    ok,bgr=c.read()
                    if not ok: break
                    for x,y,a,b in face.detectMultiScale(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),1.12,5):
                        rect=bgr[y:y+b,x:x+a]; rect[:]=cv2.GaussianBlur(rect,(0,0),max(a,b)/7)
                    stream.send(bgr)
            finally: stream.close(); c.release()
if __name__=="__main__": main(**json.loads(Path(sys.argv[1]).read_text()))
