# SPDX-License-Identifier: AGPL-3.0-only
"""Camera-space overlays. All meshes use the intrinsics from the inference result."""
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
import pyrender
import trimesh
from fab.seg_maps import full_map, paste_roi
from fab.video import BODY_LINKS, FEET
from soma._smpl_family_loader import ensure_chumpy_compat

class Layers:
    def __init__(self, K, width, height, fitted, robot):
        self.K = K
        self.renderer = pyrender.OffscreenRenderer(width,height)
        ensure_chumpy_compat()
        with (Path(os.environ["FAB_HOME"])/"body_models/smpl/SMPL_NEUTRAL.pkl").open("rb") as f:
            self.faces = pickle.load(f,encoding="latin1")["f"]
        self.fitted = fitted
        if fitted == "SOMA":
            import torch
            from soma.body import SOMALayer
            from soma.fitting.pose_inversion import PoseInversion
            self.torch=torch
            self.inv=PoseInversion(SOMALayer(Path(os.environ["SOMA_ASSETS"]),identity_model_type="smpl",identity_model_kwargs={"model_path":str(Path(os.environ["FAB_HOME"])/"body_models/smpl/SMPL_NEUTRAL.pkl")},lod="low",device="cuda",mode="warp"),low_lod=True)
        self.robot=robot
        if robot:
            import mujoco
            from fab.retarget.g1_model15 import NAMES
            self.mj=mujoco
            self.model=mujoco.MjModel.from_xml_path(str(Path(os.environ["FAB_G1"])/"g1_29dof.xml"))
            m=self.model; self.data=mujoco.MjData(m)
            self.addr=[int(m.joint(n).qposadr[0]) for n in NAMES]
            self.geoms=[]
            for i in range(m.ngeom):
                if m.geom_group[i]!=1 or m.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH: continue
                mesh=m.geom_dataid[i]; v=m.mesh_vertadr[mesh]; f=m.mesh_faceadr[mesh]
                self.geoms.append((i,m.mesh_vert[v:v+m.mesh_vertnum[mesh]],m.mesh_face[f:f+m.mesh_facenum[mesh]],m.geom_rgba[i,:3]))
    def body(self,p):
        v=p.get("mesh_smpl_fit_smoothed",p["mesh_smpl_fit"])
        if self.fitted!="SOMA": return v,self.faces
        t=self.torch
        self.inv.prepare_identity(t.as_tensor(p["smpl_betas"],dtype=t.float32,device="cuda").reshape(1,-1))
        res=self.inv.fit(t.as_tensor(v,dtype=t.float32,device="cuda")[None],body_iters=3,full_iters=1)
        s=self.inv.soma
        with t.no_grad():
            sv=s.pose(res["rotations"][:,1:],res["root_translation"],pose2rot=False,absolute_pose=True,apply_correctives=False)["vertices"]
        return sv[0].detach().cpu().numpy(),s.faces.cpu().numpy()
    def robot_meshes(self,p):
        m,d=self.model,self.data
        d.qpos[:7]=[0,0,0,1,0,0,0]
        d.qpos[self.addr]=p.get("g1_q_smoothed",p["g1_q"])
        self.mj.mj_forward(m,d)
        root=p.get("g1_root_smoothed",p["g1_root"]); loc=np.asarray(p["cam_t"]).reshape(3)
        for i,v,f,color in self.geoms:
            moved=v@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
            yield moved@root.T+loc,f,color
    def reference_meshes(self,refs,T):
        m,d=self.model,self.data
        for names,q,pos,quat,scale,dz in refs:
            d.qpos[:3]=pos; d.qpos[3:7]=quat[[3,0,1,2]]
            d.qpos[[int(m.joint(n).qposadr[0]) for n in names]]=q
            self.mj.mj_forward(m,d)
            for i,v,f,color in self.geoms:
                world=(v @ d.geom_xmat[i].reshape(3,3).T + d.geom_xpos[i]) / scale
                world[:,2]-=dz
                yield (world-T[:3,3]) @ T[:3,:3],f,color
    def render(self,bgr,objects):
        scene=pyrender.Scene(ambient_light=[.55]*3)
        gl=np.diag([1.,-1.,-1.,1.])
        K=self.K
        scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],zfar=200),pose=gl)
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=gl)
        for v,f,c in objects:
            mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
            mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*c,1.],metallicFactor=.12,roughnessFactor=.65,doubleSided=True)
            scene.add(pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False))
        rgb,z=self.renderer.render(scene)
        return np.where((z>0)[:,:,None],rgb[:,:,::-1],bgr)
    def panels(self,bgr,people,draw_robot,crop,refs,T):
        h,w=bgr.shape[:2]; parts=bgr.copy(); points=bgr.copy(); contacts=bgr.copy(); meshes=[]; robots=[]
        cmap=cv2.applyColorMap(((np.arange(256)*31)%256).astype("uint8"),cv2.COLORMAP_TURBO).reshape(256,3)
        for p in people:
            mask=p.get("parts")
            if mask is None and "parts_crop" in p:
                mask=full_map(*paste_roi(p["parts_crop"],p["crop_affine"],crop,h,w),h,w)
            if mask is None and "parts_roi" in p: mask=full_map(p["parts_roi"],p["roi"],h,w)
            if mask is not None:
                live=mask!=0; parts[live]=(.25*parts[live]+.75*cmap[mask[live]]).astype("uint8")
            kp=p.get("keypoints2d",p["keypoints2d_projected"]); c=p.get("contact_video",p.get("contact"))
            if "dense_surface2d" in p:
                depth=p.get("dense_surface_cam",p["dense_surface"])[:,2]
                lo,hi=np.percentile(depth,[5,95]); indices=np.clip((depth-lo)/max(hi-lo,1e-6)*255,0,255).astype("uint8")
                colors=cv2.applyColorMap(indices,cv2.COLORMAP_TURBO).reshape(-1,3)
                for j in np.argsort(depth)[::-1]:
                    cv2.circle(points,tuple(p["dense_surface2d"][j].astype(int)),2,tuple(map(int,colors[j])), -1,cv2.LINE_AA)
            for a,b in BODY_LINKS: cv2.line(contacts,tuple(kp[a].astype(int)),tuple(kp[b].astype(int)),(255,255,255),2,cv2.LINE_AA)
            if c is not None:
                for _,ch,k in (*FEET,("L hand",4,62),("R hand",5,41),("L knee",6,11),("R knee",7,12)):
                    cv2.circle(contacts,tuple(kp[k].astype(int)),6,(65,65,250) if c[ch]>=.5 else (145,145,145),2)
            meshes.append((*self.body(p),(.77,.84,.91)))
            if draw_robot: robots.extend(self.robot_meshes(p))
        if refs: robots=list(self.reference_meshes(refs,T))
        return (bgr,self.render(bgr,robots) if robots else bgr,self.render(bgr,meshes),parts,contacts,points)
