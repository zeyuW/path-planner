"""表面与视点预览，绘图函数只负责输出可视化文件。"""
from __future__ import annotations
from .models import SurfaceModel, Viewpoint
import numpy as np


def _write_preview(path, boundary, scan_area, selected, origin, axes) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    figure = Figure(figsize=(17, 4.2))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    colors = ['#27a5bc', '#55bda2', '#93b878', '#ceae69']
    for index, face in enumerate(selected):
        local = (face.triangles - origin) @ axes
        ax.add_collection(PolyCollection(local[:, :, :2], facecolor=colors[index % 4],
                                         edgecolor='none', antialiaseds=False,
                                         label=f'Face {face.face_id}'))
    ax.plot(*boundary.exterior.xy, color='#225461', linewidth=0.8)
    for hole in boundary.interiors:
        ax.plot(*hole.xy, color='#225461', linewidth=0.5)
    ax.plot(*scan_area.exterior.xy, color='#d84d3f', linewidth=1, label='Coverage boundary')
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('Local X / length (mm)')
    ax.set_ylabel('Local Y / width (mm)')
    ax.set_title(f'STEP target surface | {len(selected)} CAD faces | {len(boundary.interiors)} holes')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=min(len(selected)+1, 6), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def preview_surface(surface: SurfaceModel, path) -> None:
    _write_preview(path, surface.boundary, surface.scan_area, surface.preview_faces,
                   surface.local_to_world[:3, 3], surface.local_to_world[:3, :3])


def export_viewpoint_visualization(surface: SurfaceModel, viewpoints: list[Viewpoint], path) -> None:
    """导出局部表面和相机方向标记。"""
    from .file_io import _write_obj
    triangles = list(surface.triangles)
    for view in viewpoints:
        direction = view.direction / np.linalg.norm(view.direction)
        reference = np.eye(3)[np.argmin(np.abs(direction))]
        u = np.cross(direction, reference)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        apex = view.position + direction * 50
        angles = np.arange(3) * 2 * np.pi / 3
        base = view.position + 25 * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        for index in range(3):
            triangles.append(np.array([apex, base[index], base[(index+1) % 3]]))
        triangles.append(base[::-1])
    _write_obj(path, np.asarray(triangles))
