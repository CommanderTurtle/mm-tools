# SPDX-License-Identifier: AGPL-3.0-only
"""Retarget continuous tracks and map the 50 Hz kinematic results into source frames."""
from collections import defaultdict
from itertools import groupby
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def cameras(upload, count, fps):
    if not upload:
        # OpenCV x-right/y-down/z-forward to a fixed, z-up reference.
        return np.broadcast_to(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1.]]), (count,4,4)), None
    with np.load(upload, allow_pickle=False) as data:
        poses = data["T_world_cam"]
        if "fps" in data and not np.isclose(float(data["fps"]), fps, rtol=.002):
            raise ValueError("Camera frame rate does not match the uploaded footage.")
        if "frames" in data and not np.array_equal(data["frames"], np.arange(count)):
            raise ValueError("Camera frame indices must match the video, starting at zero.")
        if poses.shape != (count,4,4) or not np.isfinite(poses).all() or not np.allclose(poses[:,3], [0,0,0,1]):
            raise ValueError("Expected one finite camera-to-world 4×4 transform per video frame.")
        rot = poses[:,:3,:3]
        if not np.allclose(rot.transpose(0,2,1) @ rot,np.eye(3),atol=.002) or not np.allclose(np.linalg.det(rot),1,atol=.002):
            raise ValueError("Camera orientations must be proper rotations in a metric, z-up world.")
        K = data["K"] if "K" in data else None
        if K is not None and K.shape == (count,3,3) and np.allclose(K,K[0]): K=K[0]
        if K is not None and (K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0):
            raise ValueError("fab-video uses one calibrated intrinsic matrix for the clip.")
        return poses, K


def solve(frames, fps, poses, output, all_people):
    tracks = defaultdict(list)
    for i, frame in enumerate(frames):
        for person in frame["people"]:
            tracks[person["track"]].append((i,person))
    segments = [list(group) for people in tracks.values()
                for _,group in groupby(enumerate(people),lambda x:x[1][0]-x[0])]
    # Do not animate across a tracking gap, or join two people in one solve.
    segments = sorted((s for s in segments if len(s)>=2), key=len,reverse=True)
    replay = defaultdict(list)
    for seg in (segments if all_people else segments[:1]):
        ids, people = zip(*(item for _,item in seg))
        T = poses[list(ids)]
        verts = np.stack([p.get("mesh_smpl_fit_smoothed",p["mesh_smpl_fit"]) for p in people])
        world = verts @ T[:,:3,:3].transpose(0,2,1) + T[:,None,:3,3]
        tag=f"person-{people[0]['track']}-frame-{ids[0]}"
        stage=output/"retarget"/tag
        stage.mkdir(parents=True)
        source=stage/f"{tag}.npz"
        np.savez(source,verts=world.astype(np.float32),fps=fps,frames=ids)
        subprocess.run([Path(sys.executable).with_name("fab-retarget"),source,"--mode","mesh","--out",stage],check=True)
        result=stage/f"{tag}_g1.npz"
        info=json.loads((stage/f"{tag}_g1.json").read_text())
        dz=float(info["smpl_input"]["ground_dz_m"])
        with np.load(result,allow_pickle=False) as z:
            names=tuple(str(n) for n in z["g1_joint_names"])
            angles,positions,quats=z["g1_dof"],z["g1_root_pos"],z["g1_root_quat_xyzw"]
            scale=float(z["scale"])
            for local,idx in enumerate(ids):
                j=min(round(local*float(z["fps"])/fps),len(angles)-1)
                replay[idx].append((names,angles[j],positions[j],quats[j],scale,dz))
        if not (output/"reference.mp4").exists():
            from reference import movie
            movie(result,output/"reference.mp4")
    return replay
