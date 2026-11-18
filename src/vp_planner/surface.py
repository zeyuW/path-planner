"""通过整件 PCA 识别连续宽面，生成局部网格、边界和坐标变换。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .models import SurfaceModel


@dataclass


class FaceMesh:
    face_id: int
    shape: object
    kind: str
    triangles: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    neighbors: set[int]

    @property
    def area(self) -> float:
        return float(self.areas.sum())

    @property
    def normal(self) -> np.ndarray:
        vector = (self.normals * self.areas[:, None]).sum(axis=0)
        return vector / np.linalg.norm(vector)


def _triangle_geometry(triangles) -> tuple[np.ndarray, np.ndarray]:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    return cross / lengths[:, None], lengths / 2


def _mesh_faces(shape, linear_deflection_mm, angular_deflection_deg) -> list[FaceMesh]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_ShapeMapHasher

    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection_mm, False,
                                     float(np.radians(angular_deflection_deg)), True)
    if not mesher.IsDone():
        raise ValueError('STEP 网格生成失败')
    faces, edge_buckets = [], {}
    # Python 包装对象的 hash 不代表 CAD 拓扑身份；用 OCCT hasher + IsSame。
    hasher = TopTools_ShapeMapHasher()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        fid = len(faces) + 1
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is None or mesh.NbTriangles() == 0:
            raise ValueError(f'面 {fid} 无有效网格，不能保证完整提取')
        vertices = []
        for index in range(1, mesh.NbNodes() + 1):
            p = mesh.Node(index).Transformed(location.Transformation())
            vertices.append((p.X(), p.Y(), p.Z()))
        indices = np.array([[mesh.Triangle(i).Value(j) - 1 for j in (1, 2, 3)]
                            for i in range(1, mesh.NbTriangles() + 1)])
        triangles = np.array(vertices)[indices]
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, [0, 2, 1]]
        good = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-10
        triangles = triangles[good]
        if not len(triangles):
            raise ValueError(f'面 {fid} 的网格全部退化')
        normals, areas = _triangle_geometry(triangles)
        faces.append(FaceMesh(fid, face, str(BRepAdaptor_Surface(face).GetType()),
                              triangles, normals, areas, set()))
        edges = TopExp_Explorer(face, TopAbs_EDGE)
        while edges.More():
            edge = edges.Current()
            bucket = edge_buckets.setdefault(hasher(edge), [])
            match = next((owners for other, owners in bucket if edge.IsSame(other)), None)
            if match is None:
                bucket.append((edge, {fid}))
            else:
                match.add(fid)
            edges.Next()
        explorer.Next()
    for bucket in edge_buckets.values():
        for _, owners in bucket:
            for fid in owners:
                faces[fid - 1].neighbors.update(owners - {fid})
    return faces


def _select_patch(faces, target_direction, angle_tolerance_deg,
                  adjacency_angle_deg, seed_face_id=None) -> tuple[list[FaceMesh], int, set[int]]:
    """仅沿共享 CAD 边扩展，不靠投影重叠或强制翻转法向拼接正反面。"""
    direction = np.array(target_direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('target_direction 必须是非零三维方向')
    direction /= np.linalg.norm(direction)
    limit = np.cos(np.radians(angle_tolerance_deg))
    candidates = {f.face_id: f for f in faces if np.all(f.normals @ direction >= limit)}
    if not candidates:
        raise ValueError('没有满足指定朝向的面，请检查 target_direction 和法向容差')
    if seed_face_id is None:
        seed_face_id = max(candidates, key=lambda fid: candidates[fid].area)
    if seed_face_id not in candidates:
        raise ValueError(f'种子面 {seed_face_id} 不存在或不符合指定朝向')
    selected, stack = {seed_face_id}, [seed_face_id]
    smooth_limit = np.cos(np.radians(adjacency_angle_deg))
    while stack:
        current = candidates[stack.pop()]
        for fid in (current.neighbors & candidates.keys()) - selected:
            if np.dot(current.normal, candidates[fid].normal) >= smooth_limit:
                selected.add(fid)
                stack.append(fid)
    return [candidates[fid] for fid in sorted(selected)], seed_face_id, set(candidates)


def _surface_pca(triangles) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按三角形表面积积分，避免网格顶点密度左右主方向。"""
    _, areas = _triangle_geometry(triangles)
    origin = np.average(triangles.mean(axis=1), axis=0, weights=areas)
    centered = triangles - origin
    # 三角形均匀分布的二阶矩，避免 PCA 受三角形密度影响。
    sums = centered.sum(axis=1)
    moments = (np.einsum('nti,ntj->nij', centered, centered)
               + np.einsum('ni,nj->nij', sums, sums)) / 12
    covariance = np.average(moments, axis=0, weights=areas)
    values, vectors = np.linalg.eigh(covariance)
    return origin, values, vectors, covariance


def _select_automatic_patch(faces, angle_tolerance_deg, adjacency_angle_deg) -> tuple[list[FaceMesh], int, set[int], dict]:
    """PCA 厚度轴筛选两侧宽面，选择面积最大的连续区域。"""
    triangles = np.concatenate([face.triangles for face in faces])
    _, values, vectors, _ = _surface_pca(triangles)
    if values[1] <= 1e-12 or values[0] >= 0.8 * values[1]:
        raise ValueError('模型没有清晰的板面厚度方向，无法自动识别待测宽面')
    thickness_axis = vectors[:, 0]
    limit = np.cos(np.radians(angle_tolerance_deg))
    candidates = {face.face_id: face for face in faces
                  if np.all(face.normals @ thickness_axis >= limit)
                  or np.all(face.normals @ thickness_axis <= -limit)}
    if not candidates:
        raise ValueError('PCA 厚度方向附近没有连续宽面，请检查模型形状和角度容差')
    remaining = set(candidates)
    patches = []
    smooth_limit = np.cos(np.radians(adjacency_angle_deg))
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        patch, stack = {seed}, [seed]
        while stack:
            current = candidates[stack.pop()]
            for fid in sorted(current.neighbors & remaining):
                if np.dot(current.normal, candidates[fid].normal) >= smooth_limit:
                    remaining.remove(fid)
                    patch.add(fid)
                    stack.append(fid)
        patches.append(sorted(patch))
    areas = [sum(candidates[fid].area for fid in patch) for patch in patches]
    largest_area = max(areas)
    # 面积近似相同的区域按 CAD 面顺序决胜，不借助全局坐标正负。
    tied = [i for i, area in enumerate(areas) if area >= largest_area * (1 - 1e-5)]
    winner = min(tied, key=lambda i: patches[i][0])
    selected = [candidates[fid] for fid in patches[winner]]
    seed = max(selected, key=lambda face: face.area).face_id
    info = {'pca_variances_mm2': values.tolist(),
            'thickness_axis_world': thickness_axis.tolist(),
            'side_ambiguous': len(tied) > 1,
            'tie_break_rule': 'first CAD face among equal-area patches',
            'patches': [{'face_ids': patch, 'area_mm2': area}
                        for patch, area in zip(patches, areas)]}
    return selected, seed, set(candidates), info


def _local_frame(triangles) -> tuple[np.ndarray, np.ndarray]:
    """局部 X 沿长度、Y 沿宽度，Z 与表面朝外法向相反。"""
    normals, areas = _triangle_geometry(triangles)
    origin, _, _, covariance = _surface_pca(triangles)
    z_axis = -np.average(normals, axis=0, weights=areas)
    z_axis /= np.linalg.norm(z_axis)
    projection = np.eye(3) - np.outer(z_axis, z_axis)
    _, vectors = np.linalg.eigh(projection @ covariance @ projection)
    x_axis = vectors[:, -1]
    edges = (np.roll(triangles, -1, axis=1) - triangles).reshape(-1, 3)
    projections = edges @ x_axis
    longest = np.abs(projections).max()
    anchor = np.flatnonzero(np.abs(projections) >= longest * (1 - 1e-8))[0]
    if projections[anchor] < 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def _refine_triangles(triangles, max_edge_mm) -> np.ndarray:
    """最长边二分，控制真正的边长；保留已离散曲面的形状和朝向。"""
    pending, finished = triangles, []
    for _ in range(40):
        lengths = np.stack([np.linalg.norm(pending[:, (i+1) % 3] - pending[:, i], axis=1)
                            for i in range(3)], axis=1)
        large = lengths.max(axis=1) > max_edge_mm * (1 + 1e-12)
        finished.append(pending[~large])
        if not large.any():
            return np.concatenate(finished)
        pending = pending[large]
        longest = lengths[large].argmax(axis=1)
        rows = np.arange(len(pending))
        a, b, c = (pending[rows, (longest + j) % 3] for j in range(3))
        mid = (a + b) / 2
        pending = np.concatenate((np.stack((a, mid, c), axis=1),
                                  np.stack((mid, b, c), axis=1)))
        if len(pending) + sum(map(len, finished)) > 2_000_000:
            raise ValueError('细分网格超过 200 万片，请增大 mesh_size_mm')
    raise ValueError('网格细分未收敛')


def _project_boundary(triangles) -> Polygon:
    """合并局部 XY 投影，拒绝断开区域和明显重叠。"""
    polygons = [set_precision(Polygon(tri[:, :2]), 1e-5) for tri in triangles]
    boundary = unary_union(polygons)
    if boundary.geom_type != 'Polygon' or boundary.is_empty or not boundary.is_valid:
        raise ValueError('投影不是单个连续区域，请检查断缝或分区规划；不可只保留最大块')
    projected_area = sum(p.area for p in polygons)
    if projected_area - boundary.area > max(1.0, projected_area * 1e-4):
        raise ValueError('目标面存在明显投影重叠，二维覆盖前需要分区')
    return boundary


def prepare_surface(step_file: str | Path, config: dict) -> SurfaceModel:
    """读取 STEP 并返回局部待测表面，结果文件由调用者导出。"""
    step_file = Path(step_file)
    expansion_mm = config.get('boundary_expansion_mm', 50.0)
    angle_tolerance_deg = config.get('normal_tolerance_deg', 25.0)
    mesh_size_mm = config.get('mesh_size_mm', 40.0)
    adjacency_angle_deg = config.get('adjacency_angle_deg', 15.0)
    linear_deflection_mm = config.get('linear_deflection_mm', 0.5)
    angular_deflection_deg = config.get('angular_deflection_deg', 10.0)
    target_direction = config.get('target_direction')
    seed_face_id = config.get('seed_face_id')
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    if not all(np.isfinite(x) and x > 0 for x in (mesh_size_mm, linear_deflection_mm, angular_deflection_deg)):
        raise ValueError('网格边长、曲面误差和角度误差必须为正数')
    if not np.isfinite(expansion_mm) or expansion_mm < 0:
        raise ValueError('边界外扩距离不能为负数')
    if not (0 < angle_tolerance_deg < 90 and 0 < adjacency_angle_deg < 90):
        raise ValueError('法向角度容差必须在 0 到 90 度之间')
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_file)) != IFSelect_RetDone:
        raise ValueError(f'STEP 文件读取失败: {step_file}')
    reader.SetSystemLengthUnit(1.0)
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError('STEP 文件中没有有效几何体')
    print('STEP 已读取，正在建立面网格和共享边关系…', flush=True)
    faces = _mesh_faces(shape, linear_deflection_mm, angular_deflection_deg)
    if target_direction is None:
        if seed_face_id is not None:
            raise ValueError('自动模式不接受种子面，请移除 seed_face_id')
        selected, seed, candidate_ids, automatic_info = _select_automatic_patch(
            faces, angle_tolerance_deg, adjacency_angle_deg)
        if automatic_info['side_ambiguous']:
            print('两侧宽面面积接近，按 CAD 面顺序自动选择一侧；已在 selection.json 标记。')
    else:
        # 兼容已有直接调用者，主程序统一使用自动模式。
        selected, seed, candidate_ids = _select_patch(
            faces, target_direction, angle_tolerance_deg, adjacency_angle_deg, seed_face_id)
        automatic_info = None
    selected_ids = {f.face_id for f in selected}
    triangles = np.concatenate([f.triangles for f in selected])
    origin, axes = _local_frame(triangles)
    local = (triangles - origin) @ axes
    boundary = _project_boundary(local)
    # 仅外轮廓外扩，孔洞保持原尺寸。直接 buffer 带孔区域会缩小甚至填掉孔。
    expanded_outer = Polygon(boundary.exterior).buffer(expansion_mm, join_style='mitre')
    scan_area = Polygon(expanded_outer.exterior, [ring.coords for ring in boundary.interiors])
    refined = _refine_triangles(local, mesh_size_mm)
    normals, _ = _triangle_geometry(refined)
    if np.any(normals[:, 2] >= 0):
        raise ValueError('目标面不适合统一的局部投影方向，需要分区规划')
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    for face in selected:
        builder.Add(compound, face.shape)
    local_to_world = np.eye(4)
    local_to_world[:3, :3], local_to_world[:3, 3] = axes, origin
    manifest = {
        'source': str(step_file.resolve()), 'units': 'mm', 'local_plane': 'XY',
        'selection_rule': 'PCA thickness + largest connected broad patch' if automatic_info else 'explicit direction',
        'automatic_selection': automatic_info,
        'seed_face_id': seed, 'selected_face_ids': sorted(selected_ids),
        'total_faces': len(faces), 'candidate_face_count': len(candidate_ids),
        'triangle_count': len(refined), 'surface_area_mm2': sum(f.area for f in selected),
        'hole_count': len(boundary.interiors), 'boundary_bounds_xy_mm': list(boundary.bounds),
        'depth_range_local_z_mm': [float(local[:, :, 2].min()), float(local[:, :, 2].max())],
        'mesh_max_edge_mm': mesh_size_mm, 'linear_deflection_mm': linear_deflection_mm,
        'angular_deflection_deg': angular_deflection_deg, 'boundary_expansion_mm': expansion_mm,
        'target_direction': list(target_direction) if target_direction is not None else None,
        'normal_tolerance_deg': angle_tolerance_deg,
        'adjacency_angle_deg': adjacency_angle_deg,
        'local_to_world': local_to_world.tolist(), 'world_to_local': np.linalg.inv(local_to_world).tolist(),
        'coordinate_note': 'column vectors; local X is length, Y is width, Z opposes outward surface normal',
        'faces': [{'id': f.face_id, 'type': f.kind, 'area_mm2': f.area,
                   'selected': f.face_id in selected_ids, 'direction_candidate': f.face_id in candidate_ids,
                   'neighbors': sorted(f.neighbors)} for f in faces],
    }
    print(f"选中面 {sorted(selected_ids)}，孔洞 {len(boundary.interiors)} 个，三角面 {len(refined)} 个")
    return SurfaceModel(refined, normals, boundary, scan_area, local_to_world,
                        manifest, compound, selected)
