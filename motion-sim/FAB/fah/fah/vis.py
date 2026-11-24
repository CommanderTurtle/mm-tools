import cv2
import numpy as np

from .metadata import FINGER_COLORS, HAND_SKELETON

RIGHT_COLOR = np.array([0.98, 0.70, 0.45])  # BGR in [0, 1], light blue
LEFT_COLOR = np.array([0.55, 0.80, 0.98])  # light orange


def project(points, cam_int):
    uv = points[:, :2] / np.clip(points[:, 2:3], 1e-6, None)
    return uv * cam_int[[0, 1], [0, 1]] + cam_int[[0, 1], [2, 2]]


def render_meshes(img, meshes, cam_int, alpha=0.9, supersample=2):
    """Painter's-algorithm mesh rendering with Lambert shading (no OpenGL needed).
    meshes: list of (vertices (V, 3) in camera frame, faces (F, 3), base color BGR 0~1)."""
    h, w = img.shape[:2]
    s = supersample
    K = cam_int.copy()
    K[:2] *= s
    canvas = cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((h * s, w * s), dtype=np.uint8)

    polys, depths, colors = [], [], []
    for verts, faces, color in meshes:
        if len(faces) == 0 or (verts[:, 2] <= 1e-3).any():
            continue
        uv = project(verts, K)
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        centers = tri.mean(axis=1)
        view = -centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
        shade = np.abs((normals * view).sum(axis=1))
        shade = 0.25 + 0.75 * shade
        polys.append(uv[faces])
        depths.append(centers[:, 2])
        colors.append(shade[:, None] * np.asarray(color)[None])
    if not polys:
        return img.copy()

    polys = np.concatenate(polys)
    depths = np.concatenate(depths)
    colors = (np.concatenate(colors) * 255).clip(0, 255)
    order = np.argsort(-depths)
    pts = np.round(polys * 16).astype(np.int32)  # 4-bit subpixel precision
    for i in order:
        c = colors[i]
        cv2.fillConvexPoly(canvas, pts[i], (float(c[0]), float(c[1]), float(c[2])), cv2.LINE_8, 4)
        cv2.fillConvexPoly(mask, pts[i], 255, cv2.LINE_8, 4)

    canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)[..., None] / 255.0
    out = img * (1 - alpha * mask) + canvas * (alpha * mask)
    return out.astype(np.uint8)


def draw_hand_skeleton(img, kp2d, thickness=2, radius=3):
    img = img.copy()
    pts = np.round(kp2d).astype(int)
    for e, (a, b) in enumerate(HAND_SKELETON):
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), FINGER_COLORS[e // 4], thickness, cv2.LINE_AA)
    for j, p in enumerate(pts):
        color = (255, 255, 255) if j == 0 else FINGER_COLORS[(j - 1) // 4]
        cv2.circle(img, tuple(p), radius, color, -1, cv2.LINE_AA)
    return img


def draw_boxes(img, hands):
    img = img.copy()
    for hand in hands:
        x1, y1, x2, y2 = hand["bbox"].astype(int)
        color = (255, 128, 0) if hand["is_right"] else (0, 128, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{'R' if hand['is_right'] else 'L'} {hand['score']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def side_view_meshes(hands, angle_deg=90.0):
    """Rotate each hand about the vertical axis through its own center (shows depth
    and finger articulation; hands at very different depths stay in view)."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    meshes = []
    for h in hands:
        center = (h["vertices"].max(0) + h["vertices"].min(0)) / 2
        v = (h["vertices"] - center) @ R.T + center
        meshes.append((v, h["faces"], RIGHT_COLOR if h["is_right"] else LEFT_COLOR))
    return meshes


def visualize_hands(img_bgr, hands, side_view=True):
    """Returns [input+boxes | 2D keypoints | mesh overlay | side view]."""
    panels = [draw_boxes(img_bgr, hands)]
    kp_img = img_bgr.copy()
    for hand in hands:
        kp_img = draw_hand_skeleton(kp_img, hand["keypoints_2d"])
    panels.append(kp_img)
    if not hands:
        panels.append(img_bgr.copy())
        if side_view:
            panels.append(np.full_like(img_bgr, 255))
        return np.concatenate(panels, axis=1)

    cam_int = hands[0]["cam_int"]
    meshes = [
        (h["vertices"], h["faces"], RIGHT_COLOR if h["is_right"] else LEFT_COLOR)
        for h in hands
    ]
    panels.append(render_meshes(img_bgr, meshes, cam_int))
    if side_view:
        white = np.full_like(img_bgr, 255)
        panels.append(render_meshes(white, side_view_meshes(hands), cam_int, alpha=1.0))
    return np.concatenate(panels, axis=1)


def render_instance_masks(meshes, cam_int, hw):
    """Visible-instance id map (0 = background, k = k-th mesh) by painter's algorithm over all meshes'
    triangles (mutual hand occlusion handled; scene objects are not modelled).
    meshes: list of (vertices (V, 3) camera frame, faces (F, 3))."""
    h, w = hw
    ids = np.zeros((h, w), dtype=np.uint8)
    polys, depths, inst = [], [], []
    for k, (verts, faces) in enumerate(meshes):
        if len(faces) == 0 or (verts[:, 2] <= 1e-3).any():
            continue
        tri = verts[faces]
        polys.append(project(verts, cam_int)[faces])
        depths.append(tri[:, :, 2].mean(1))
        inst.append(np.full(len(faces), k + 1, np.uint8))
    if not polys:
        return ids
    polys, depths, inst = np.concatenate(polys), np.concatenate(depths), np.concatenate(inst)
    pts = np.round(polys * 16).astype(np.int32)
    for i in np.argsort(-depths):
        cv2.fillConvexPoly(ids, pts[i], int(inst[i]), cv2.LINE_8, 4)
    return ids
