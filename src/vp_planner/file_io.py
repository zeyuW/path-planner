"""读取 OBJ 网格和扫描区域，按三角形中心筛选待测面。

三角面行格式为编号、三个顶点 XXY、法向量。
区域行格式为编号、点数、XY 坐标，首行为外轮廓，其余为孔洞。"""
from __future__ import annotations

from .models import SurfaceModel, Viewpoint, PlanningResult
from shapely.geometry.base import BaseGeometry
from pathlib import Path
import json
import os
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import unary_union


def parse_obj(file_path) -> tuple[np.ndarray, np.ndarray]:
    """读取 OBJ 顶点和三角面，索引转为从 0 开始。"""
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_vertices = []
                valid_face = True
                for part in parts[1:]:
                    indices = part.split('/')
                    if not indices[0].strip().isdigit():
                        valid_face = False
                        break
                    vertex_index = int(indices[0]) - 1
                    if vertex_index < 0 or vertex_index >= len(vertices):
                        valid_face = False
                        break
                    face_vertices.append(vertex_index)
                if valid_face and len(face_vertices) == 3:
                    faces.append(face_vertices)
    print(f"OBJ文件解析完成: 顶点数={len(vertices)}, 有效三角面片数={len(faces)}")
    return np.array(vertices), np.array(faces)


def calculate_normal(p1, p2, p3) -> np.ndarray:
    """计算三角面片的法向量，并调整方向朝向局部 Z 轴负方向"""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal = normal / norm
    else:
        normal = np.zeros(3)

    if normal[2] > 0:
        normal = -normal
    return normal


def discretize_3d_model(obj_path, txt_path) -> bool:
    """将已有 OBJ 三角网格转为规划器使用的 TXT 格式。"""
    print("第一步：三维模型离散化")
    if not os.path.exists(obj_path):
        print(f"错误：OBJ文件不存在: {obj_path}")
        return False

    vertices, faces = parse_obj(obj_path)
    if len(faces) == 0:
        print("错误：未找到有效的三角面片")
        return False

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, 'w') as out_file:
        for i, face in enumerate(faces):
            p1 = vertices[face[0]]
            p2 = vertices[face[1]]
            p3 = vertices[face[2]]
            normal = calculate_normal(p1, p2, p3)

            line = f"{i+1} "
            line += " ".join(f"{x:.6f}" for x in p1) + " "
            line += " ".join(f"{x:.6f}" for x in p2) + " "
            line += " ".join(f"{x:.6f}" for x in p3) + " "
            line += " ".join(f"{x:.6f}" for x in normal)
            out_file.write(line + "\n")

    print(f"离散化完成！共生成 {len(faces)} 个三角面片")
    return True


def parse_area_file(file_path: str | Path) -> tuple[BaseGeometry, BaseGeometry, BaseGeometry | None]:
    """区域首行为外轮廓，其余行是孔洞，坐标均为局部 XY。"""
    polygons = []
    with Path(file_path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.split()
            count = int(parts[1])
            if count < 3 or len(parts) != 2 + 2 * count:
                raise ValueError('区域轮廓的点数或坐标数量不正确')
            coords = np.asarray(parts[2:], dtype=float).reshape(-1, 2)
            polygon = ShapelyPolygon(coords)
            polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    if not polygons:
        raise ValueError('区域文件为空')
    outer = polygons[0]
    holes = unary_union(polygons[1:]) if len(polygons) > 1 else None
    effective = outer.difference(holes) if holes is not None else outer
    return effective, outer, holes


def filter_triangles_by_area(surface_file, outer_boundary, holes, output_file) -> int:
    """按面片中心的 XY 投影筛选，保留原三维坐标和法向。"""
    rows = np.loadtxt(surface_file, ndmin=2)
    if rows.shape[1] != 13:
        raise ValueError('三角面文件每行应包含 13 列')
    centers = rows[:, 1:10].reshape(-1, 3, 3).mean(axis=1)
    area = outer_boundary.difference(holes) if holes is not None else outer_boundary
    selected = np.array([area.contains(Point(center[:2])) for center in centers], dtype=bool)
    np.savetxt(output_file, rows[selected], fmt=['%d'] + ['%.8f'] * 12)
    return int(selected.sum())


def filter_surface_by_area(area_file, surface_file, remaining_triangles_file) -> tuple[BaseGeometry, BaseGeometry, BaseGeometry | None, Path]:
    """兼容区域 TXT 筛选调用。"""
    effective, outer, holes = parse_area_file(area_file)
    output = Path(remaining_triangles_file)
    filter_triangles_by_area(surface_file, outer, holes, output)
    return effective, outer, holes


def _write_rings(path, polygon) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for index, ring in enumerate([polygon.exterior, *polygon.interiors], 1):
            coords = list(ring.coords)[:-1]
            stream.write(f'{index} {len(coords)} ' + ' '.join(
                f'{u:.8f} {v:.8f}' for u, v in coords) + '\n')


def _write_obj(path, triangles) -> None:
    with path.open('w', encoding='utf-8') as stream:
        stream.write('# STEP target surface; units: mm\n')
        for point in triangles.reshape(-1, 3):
            stream.write('v ' + ' '.join(f'{v:.8f}' for v in point) + '\n')
        for i in range(len(triangles)):
            stream.write(f'f {3*i+1} {3*i+2} {3*i+3}\n')


def write_triangles(path, triangles, normals) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((np.arange(1, len(triangles) + 1),
                            triangles.reshape(-1, 9), normals))
    np.savetxt(path, data, fmt=['%d'] + ['%.8f'] * 12)


def write_viewpoints(path, viewpoints: list[Viewpoint]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for index, view in enumerate(viewpoints, 1):
            values = [view.angle, *view.position, *view.direction]
            stream.write(str(index) + ' ' + ' '.join(f'{v:.8f}' for v in values) + '\n')


def export_surface(surface: SurfaceModel, surface_file=None, area_file=None, extraction_dir=None) -> None:
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    extraction_dir = Path(extraction_dir)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    if surface_file is not None:
        write_triangles(surface_file, surface.triangles, surface.normals)
    if area_file is not None:
        _write_rings(Path(area_file), surface.scan_area)
    _write_rings(extraction_dir / 'surface_boundary.txt', surface.boundary)
    transform = surface.local_to_world
    world = surface.triangles @ transform[:3, :3].T + transform[:3, 3]
    _write_obj(extraction_dir / 'target_surface_world.obj', world)
    writer = STEPControl_Writer()
    if writer.Transfer(surface.cad_shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise ValueError('提取面 STEP 转换失败')
    if writer.Write(str(extraction_dir / 'target_surface.step')) != IFSelect_RetDone:
        raise ValueError('提取面 STEP 导出失败')
    write_json(extraction_dir / 'selection.json', surface.metadata)


def write_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def export_planning_result(result: PlanningResult, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_viewpoints(output_dir / 'VP.txt', result.viewpoints)
    write_json(output_dir / 'planning_report.json', {
        'stop_reason': result.stop_reason, 'coverage_rate': result.coverage_rate,
        'remaining_triangle_count': len(result.remaining_triangles),
        'viewpoint_count': len(result.viewpoints), 'iterations': result.iterations,
        'coverage_method': 'triangle vertices in frustum and viewing angle; no occlusion test',
    })


def prepare_step_model(step_file, surface_file, area_file, expansion_mm=50.0,
                       angle_tolerance_deg=25.0, mesh_size_mm=40.0, *,
                       target_direction=None, seed_face_id=None, adjacency_angle_deg=15.0,
                       linear_deflection_mm=0.5, angular_deflection_deg=10.0,
                       extraction_dir=None) -> bool:
    """兼容文件式前处理调用。"""
    from .surface import prepare_surface
    from .visualization import preview_surface
    surface = prepare_surface(Path(step_file), {
        'boundary_expansion_mm': expansion_mm, 'normal_tolerance_deg': angle_tolerance_deg,
        'mesh_size_mm': mesh_size_mm, 'target_direction': target_direction,
        'seed_face_id': seed_face_id, 'adjacency_angle_deg': adjacency_angle_deg,
        'linear_deflection_mm': linear_deflection_mm,
        'angular_deflection_deg': angular_deflection_deg,
    })
    extraction_dir = Path(extraction_dir) if extraction_dir else Path(surface_file).parent / 'extraction'
    export_surface(surface, surface_file, area_file, extraction_dir)
    preview_surface(surface, extraction_dir / 'target_surface_preview.png')
    return True
