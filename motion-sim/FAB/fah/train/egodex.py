"""EgoDex (apple-aiml-research/ml-egodex) episode loader: ARKit hand joints in the OpenCV camera frame +
the paired RGB video. Its 2D projections are off by 5-15% of hand size (RGB/pose mismatch), so only the
3D (root-relative) labels are used for training."""
import os
import glob
import warnings
import numpy as np
import cv2
import h5py

# EgoDex (ARKit visionOS HandSkeleton) joint names -> OpenPose/MANO 21-keypoint order.
# ARKit naming (verified from bone lengths): thumbKnuckle=CMC, thumbIntermediateBase=MCP,
# thumbIntermediateTip=IP; finger Metacarpal=CMC (dropped), Knuckle=MCP,
# IntermediateBase=PIP, IntermediateTip=DIP.
OPENPOSE_SUFFIXES = [
    'Hand',                                                                                   # 0 wrist
    'ThumbKnuckle', 'ThumbIntermediateBase', 'ThumbIntermediateTip', 'ThumbTip',              # 1-4
    'IndexFingerKnuckle', 'IndexFingerIntermediateBase', 'IndexFingerIntermediateTip', 'IndexFingerTip',      # 5-8
    'MiddleFingerKnuckle', 'MiddleFingerIntermediateBase', 'MiddleFingerIntermediateTip', 'MiddleFingerTip',  # 9-12
    'RingFingerKnuckle', 'RingFingerIntermediateBase', 'RingFingerIntermediateTip', 'RingFingerTip',          # 13-16
    'LittleFingerKnuckle', 'LittleFingerIntermediateBase', 'LittleFingerIntermediateTip', 'LittleFingerTip',  # 17-20
]
OPENPOSE_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'little_mcp', 'little_pip', 'little_dip', 'little_tip',
]
OPENPOSE_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
                  (0, 5), (5, 6), (6, 7), (7, 8),
                  (0, 9), (9, 10), (10, 11), (11, 12),
                  (0, 13), (13, 14), (14, 15), (15, 16),
                  (0, 17), (17, 18), (18, 19), (19, 20)]

DEFAULT_K = np.array([[736.6339, 0., 960.], [0., 736.6339, 540.], [0., 0., 1.]])


def joint_keys(side):
    return [side + s for s in OPENPOSE_SUFFIXES]


def project(pts_cam, K):
    z = pts_cam[..., 2:3]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = pts_cam[..., :2] / zs
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


class EgoDexEpisode:
    """One EgoDex episode (paired N.hdf5 / N.mp4).

    Camera frame is OpenCV convention (x right, y down, z forward), metres.
    transforms/camera is T_world_cam (camera-to-ARKit-origin); joints_cam = inv(T_world_cam) @ T_world_joint.
    """

    def __init__(self, hdf5_path, mp4_path=None, conf_thresh=0.3, min_depth=0.05,
                 min_joints_in_img=6, bbox_pad=0.15, cam_offset=(0.0, 0.0, 0.0)):
        self.hdf5_path = hdf5_path
        self.mp4_path = mp4_path or os.path.splitext(hdf5_path)[0] + '.mp4'
        self.conf_thresh = conf_thresh
        self.min_depth = min_depth
        self.min_joints_in_img = min_joints_in_img
        self.bbox_pad = bbox_pad
        # optional virtual-camera offset (camera-frame metres) to compensate the RGB/pose perspective mismatch; default none
        self.cam_offset = np.asarray(cam_offset, dtype=np.float64)
        with h5py.File(hdf5_path, 'r') as f:
            self.K = f['camera/intrinsic'][:].astype(np.float64) if 'camera/intrinsic' in f else DEFAULT_K.copy()
            self.T_world_cam = f['transforms/camera'][:].astype(np.float64)
            T_cam_world = np.linalg.inv(self.T_world_cam)
            self.has_conf = 'confidences' in f
            self.joints_cam, self.conf = {}, {}
            for side in ('left', 'right'):
                pw = np.stack([f['transforms/' + k][:, :3, 3] for k in joint_keys(side)], 1).astype(np.float64)
                self.joints_cam[side] = np.einsum('nij,nkj->nki', T_cam_world[:, :3, :3], pw) + T_cam_world[:, None, :3, 3] - self.cam_offset
                if self.has_conf:
                    self.conf[side] = np.stack([f['confidences/' + k][:] for k in joint_keys(side)], 1).astype(np.float32)
                else:
                    self.conf[side] = None
            self.attrs = {k: f.attrs[k] for k in f.attrs}
        self.task_dir = os.path.basename(os.path.dirname(hdf5_path))
        self.episode_id = os.path.splitext(os.path.basename(hdf5_path))[0]
        self.task = str(self.attrs.get('task', self.task_dir))
        self.n_pose = self.T_world_cam.shape[0]
        cap = cv2.VideoCapture(self.mp4_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if self.n_video != self.n_pose:
            warnings.warn('%s: video frames %d != pose frames %d; using min' % (self.mp4_path, self.n_video, self.n_pose))
        self.n = min(self.n_video, self.n_pose) if self.n_video > 0 else self.n_pose

    def __len__(self):
        return self.n

    def hand_annotation(self, i, side):
        K = self.K
        j3d = self.joints_cam[side][i]
        j2d = project(j3d, K)
        in_front = j3d[:, 2] > self.min_depth
        in_img = in_front & (j2d[:, 0] >= 0) & (j2d[:, 0] < self.width) & (j2d[:, 1] >= 0) & (j2d[:, 1] < self.height)
        conf = None if self.conf[side] is None else self.conf[side][i]
        # wrist conf = whole-hand detection; finger confs are relative to the wrist (EgoDex paper App. A.2)
        wrist_ok = True if conf is None else bool(conf[0] >= self.conf_thresh)
        conf_ok = np.ones(21, bool) if conf is None else conf >= self.conf_thresh
        joint_valid = in_img & conf_ok & wrist_ok
        hand_valid = bool(wrist_ok and in_front.all() and in_img.sum() >= self.min_joints_in_img)
        bbox = None
        if in_front.all():
            x0, y0 = j2d.min(0)
            x1, y1 = j2d.max(0)
            s = max(x1 - x0, y1 - y0)
            p = self.bbox_pad * s
            x0, y0, x1, y1 = x0 - p, y0 - p, x1 + p, y1 + p
            bbox = np.array([max(x0, 0), max(y0, 0), min(x1, self.width - 1), min(y1, self.height - 1)])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bbox = None
                hand_valid = False
        else:
            hand_valid = False
        return {
            'is_right': side == 'right',
            'joints_cam': j3d.astype(np.float32),
            'joints_2d': j2d.astype(np.float32),
            'bbox': None if bbox is None else bbox.astype(np.float32),
            'conf': conf,
            'joint_in_img': in_img,
            'joint_valid': joint_valid,
            'hand_valid': hand_valid,
        }

    def frames(self, start=0, stop=None, step=1, load_image=True):
        stop = self.n if stop is None else min(stop, self.n)
        cap = cv2.VideoCapture(self.mp4_path) if load_image else None
        cur = 0
        try:
            for i in range(start, stop, step):
                img = None
                if load_image:
                    while cur < i:
                        if not cap.grab():
                            return
                        cur += 1
                    ok, img = cap.read()
                    if not ok:
                        return
                    cur += 1
                yield {
                    'frame_idx': i,
                    'image': img,
                    'K': self.K.copy(),
                    'T_world_cam': self.T_world_cam[i],
                    'hands': [self.hand_annotation(i, s) for s in ('left', 'right')],
                }
        finally:
            if cap is not None:
                cap.release()


def list_episodes(root):
    return sorted(glob.glob(os.path.join(root, '**', '*.hdf5'), recursive=True))


def draw_hands(img, hands, K=None, thickness=2):
    out = img.copy()
    cols = {True: (0, 0, 255), False: (255, 128, 0)}
    for h in hands:
        c = cols[h['is_right']]
        j = h['joints_2d']
        if not (h['joints_cam'][:, 2] > 0).all():
            continue
        for a, b in OPENPOSE_BONES:
            cv2.line(out, tuple(int(v) for v in j[a]), tuple(int(v) for v in j[b]), c, thickness, cv2.LINE_AA)
        for k in range(21):
            jc = (0, 255, 0) if h['joint_valid'][k] else (128, 128, 128)
            cv2.circle(out, tuple(int(v) for v in j[k]), 4, jc, -1, cv2.LINE_AA)
        if h['bbox'] is not None:
            x0, y0, x1, y1 = [int(v) for v in h['bbox']]
            cv2.rectangle(out, (x0, y0), (x1, y1), c if h['hand_valid'] else (128, 128, 128), 2)
            cv2.putText(out, ('R' if h['is_right'] else 'L') + (' ok' if h['hand_valid'] else ' inv'),
                        (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('hdf5')
    ap.add_argument('--frames', type=int, nargs='+', default=[0])
    ap.add_argument('--out', default='/tmp/fah_egodex/vis')
    ap.add_argument('--offset', type=int, default=0, help='pose index offset relative to video frame (debug)')
    a = ap.parse_args()
    ep = EgoDexEpisode(a.hdf5)
    want = set(a.frames)
    for fr in ep.frames(start=min(want), stop=max(want) + 1):
        if fr['frame_idx'] not in want:
            continue
        pi = int(np.clip(fr['frame_idx'] + a.offset, 0, len(ep) - 1))
        hands = [ep.hand_annotation(pi, s) for s in ('left', 'right')]
        vis = draw_hands(fr['image'], hands)
        cv2.putText(vis, '%s f%d off%d' % (ep.task_dir, fr['frame_idx'], a.offset), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        p = '%s_%s_%s_f%04d%s.jpg' % (a.out, ep.task_dir, ep.episode_id, fr['frame_idx'], '' if a.offset == 0 else '_off%+d' % a.offset)
        cv2.imwrite(p, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(p, [(h['is_right'], h['hand_valid'], int(h['joint_valid'].sum())) for h in hands])
