import os
from collections import OrderedDict

import numpy as np
import torch

FACE_NORMALS = np.asarray([
    (-1., 0., 0.), (1., 0., 0.), (0., -1., 0.),
    (0., 1., 0.), (0., 0., -1.), (0., 0., 1.)
], dtype=np.float32)


def _surface_points(points, center, size):
    lower, upper = center - size / 2., center + size / 2.
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    candidates = points[inside]
    face_centers = center[None] + FACE_NORMALS * (size[None] / 2.)
    if len(candidates) == 0:
        return face_centers
    distances = np.abs(candidates[:, None] - face_centers[None]).sum(axis=2)
    return candidates[distances.argmin(axis=0)]


def _farthest_sample(points, center, max_points):
    if len(points) <= max_points:
        return points
    selected = np.empty(max_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    selected[0] = np.linalg.norm(points - center[None], axis=1).argmax()
    for i in range(1, max_points):
        d = np.sum((points - points[selected[i - 1]]) ** 2, axis=1)
        distances = np.minimum(distances, d)
        distances[selected[:i]] = -1
        selected[i] = distances.argmax()
    return points[selected]


class BBoxKeypointProjector:
    def __init__(self, points_root='/root/shared-nvme/data/ScanNet_processed/points',
                 max_points=50, cache_size=8):
        self.points_root = points_root
        self.max_points = max_points
        self.cache_size = cache_size
        self._points_cache = OrderedDict()

    def load_points(self, lidar_path, axis_align_matrix):
        if lidar_path is None:
            return None
        if isinstance(lidar_path, (list, tuple)):
            lidar_path = lidar_path[0] if lidar_path else None
        if lidar_path is None:
            return None
        path = os.fspath(lidar_path)
        if not os.path.isabs(path):
            path = os.path.join(self.points_root, os.path.basename(path))
        key = (path, np.asarray(axis_align_matrix).tobytes())
        if key in self._points_cache:
            points = self._points_cache.pop(key)
            self._points_cache[key] = points
            return points
        if not os.path.isfile(path):
            return None
        raw = np.fromfile(path, dtype=np.float32)
        dim = 6 if raw.size % 6 == 0 else 3
        points = raw.reshape(-1, dim)[:, :3]
        axis = np.asarray(axis_align_matrix, dtype=np.float32)
        points = points @ axis[:3, :3].T + axis[:3, 3]
        self._points_cache[key] = points
        while len(self._points_cache) > self.cache_size:
            self._points_cache.popitem(last=False)
        return points

    @staticmethod
    def _views(value, num_views, dtype, device):
        value = torch.as_tensor(value, dtype=dtype, device=device)
        return value.unsqueeze(0).expand(num_views, -1, -1) if value.dim() == 2 else value

    def __call__(self, centers, dimensions, extrinsics, intrinsic, scale_factor,
                 image_shape, valid_image_shape, points=None):
        device, dtype = centers.device, centers.dtype
        n, views = centers.shape[0], len(extrinsics)
        ext = self._views(extrinsics, views, dtype, device)
        K = self._views(intrinsic, views, dtype, device)[:, :3, :3].clone()
        sf = torch.as_tensor(scale_factor, dtype=dtype, device=device)
        if sf.numel() == 2:
            sf = sf.reshape(1, 2).expand(views, 2)
        elif sf.dim() == 1:
            if sf.numel() >= 2:
                sf = sf[:2].reshape(1, 2).expand(views, 2)
            else:
                sf = sf.reshape(-1, 1).expand(-1, 2)
        else:
            sf = sf[..., :2]
        if sf.shape[0] != views:
            sf = sf[:1].expand(views, 2)
        K[:, 0] *= sf[:, 0, None]
        K[:, 1] *= sf[:, 1, None]
        image_shapes = np.asarray(image_shape)
        if image_shapes.ndim == 1:
            image_shapes = np.broadcast_to(image_shapes[None, :], (views, image_shapes.shape[0]))
        valid_shapes = np.asarray(valid_image_shape)
        if valid_shapes.ndim == 1:
            valid_shapes = np.broadcast_to(valid_shapes[None, :], (views, valid_shapes.shape[0]))
        normals = centers.new_tensor(FACE_NORMALS)
        key3d = centers.new_zeros((n, 4, 3)); key2d = centers.new_zeros((n, views, 4, 2))
        visible = torch.zeros((n, views, 4), dtype=torch.bool, device=device)
        bboxes = centers.new_zeros((n, views, 4)); bbox_visible = torch.zeros((n, views), dtype=torch.bool, device=device)
        points_np = None if points is None else np.asarray(points)
        for obj in range(n):
            c = centers[obj].detach().cpu().numpy().astype(np.float32)
            size = dimensions[obj].detach().cpu().numpy().astype(np.float32)
            if points_np is None:
                face = c[None] + FACE_NORMALS * size[None] / 2.; sampled = face
            else:
                lower, upper = c - size / 2., c + size / 2.
                inside = np.all((points_np >= lower) & (points_np <= upper), axis=1)
                sampled = _farthest_sample(points_np[inside], c, self.max_points)
                face = _surface_points(points_np, c, size)
            key3d[obj, 0] = centers[obj]
            for view in range(views):
                h, w = map(float, image_shapes[view][:2])
                vh, vw = map(float, valid_shapes[view][:2])
                e = ext[view]; cam = -e[:3, :3].transpose(0, 1) @ e[:3, 3]
                scores = torch.sum(normals * (cam[None] - centers[obj]), dim=1)
                face_t = torch.as_tensor(face, dtype=dtype, device=device)
                fc = torch.einsum('ij,fj->fi', e[:3, :3], face_t) + e[:3, 3]
                fd = fc[:, 2]; fp = torch.einsum('ij,fj->fi', K[view], fc)[:, :2] / fd[:, None].clamp_min(1e-5)
                fok = (fd > 1e-5) & torch.isfinite(fp).all(1)
                fok &= (fp[:, 0] >= 0) & (fp[:, 0] < vw) & (fp[:, 1] >= 0) & (fp[:, 1] < vh)
                order = torch.argsort(scores, descending=True); valid_order = order[fok[order]]
                selected = torch.cat([valid_order, order[~torch.isin(order, valid_order)]])[:3]
                key3d[obj, 1:] = face_t[selected]
                kc = torch.einsum('ij,fj->fi', e[:3, :3], key3d[obj]) + e[:3, 3]
                depth = kc[:, 2]; pix = torch.einsum('ij,fj->fi', K[view], kc)[:, :2] / depth[:, None].clamp_min(1e-5)
                ok = (depth > 1e-5) & torch.isfinite(pix).all(1)
                ok &= (pix[:, 0] >= 0) & (pix[:, 0] < vw) & (pix[:, 1] >= 0) & (pix[:, 1] < vh)
                key2d[obj, view] = pix / pix.new_tensor([w, h]); visible[obj, view] = ok
                if len(sampled):
                    st = torch.as_tensor(sampled, dtype=dtype, device=device)
                    pc = torch.einsum('ij,pj->pi', e[:3, :3], st) + e[:3, 3]
                    pd = pc[:, 2]; pp = torch.einsum('ij,pj->pi', K[view], pc)[:, :2] / pd[:, None].clamp_min(1e-5)
                    pv = (pd > 1e-5) & torch.isfinite(pp).all(1)
                    pv &= (pp[:, 0] >= 0) & (pp[:, 0] < vw) & (pp[:, 1] >= 0) & (pp[:, 1] < vh)
                    if pv.any() and ok[0]:
                        lo, hi = pp[pv].amin(0), pp[pv].amax(0)
                        if hi[0] > lo[0] and hi[1] > lo[1]:
                            bboxes[obj, view] = torch.cat([lo, hi]) / bboxes.new_tensor([w, h, w, h]); bbox_visible[obj, view] = True
        return key3d, key2d, visible, bboxes, bbox_visible
