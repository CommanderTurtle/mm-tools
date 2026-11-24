# SPDX-License-Identifier: AGPL-3.0-only
import os
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np

def movie(reference,output):
    model=mujoco.MjModel.from_xml_path(str(Path(os.environ["FAB_G1"])/"g1_29dof.xml"))
    z=np.load(reference,allow_pickle=False)
    data=mujoco.MjData(model); addr=[int(model.joint(str(n)).qposadr[0]) for n in z["g1_joint_names"]]
    mcam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(mcam)
    mcam.type=mujoco.mjtCamera.mjCAMERA_FREE;mcam.distance=3.4;mcam.azimuth=120;mcam.elevation=-15
    opts=mujoco.MjvOption();opts.geomgroup[:]=0;opts.geomgroup[1]=1
    width,height=640,480
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth);model.vis.global_.offheight=max(height,model.vis.global_.offheight)
    writer=imageio_ffmpeg.write_frames(str(output),(width,height),fps=50)
    writer.send(None)
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            for q,pos,quat in zip(z["g1_dof"],z["g1_root_pos"],z["g1_root_quat_xyzw"],strict=True):
                data.qpos[:3]=pos;data.qpos[3:7]=quat[[3,0,1,2]];data.qpos[addr]=q
                mujoco.mj_forward(model,data);mcam.lookat[:]=pos
                renderer.update_scene(data,camera=mcam,scene_option=opts);writer.send(renderer.render())
    finally: writer.close()
