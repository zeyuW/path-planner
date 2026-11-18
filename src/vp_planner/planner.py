"""在内存中完成二维覆盖、三维映射和漏扫补点。"""
from __future__ import annotations
import heapq
import math
import time
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon as ShapelyPolygon, Point
from .models import SurfaceModel, Viewpoint, PlanningResult


class ViewpointPlanner:
    """管理规划参数和补点循环，算法阶段直接传递数组。"""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        for name in ('rectangle_width_mm', 'rectangle_height_mm', 'viewpoint_offset_mm',
                     'candidate_grid_size_mm', 'dbscan_eps_mm'):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f'{name} 必须为正数')
        if not 0 < self.config['scale_factor'] <= 1:
            raise ValueError('scale_factor 必须在 0 到 1 之间')
        for name in ('angle_step_deg', 'max_cover_iterations', 'max_refinement_cycles',
                     'triangle_threshold', 'dbscan_min_samples'):
            if not isinstance(self.config[name], int) or self.config[name] <= 0:
                raise ValueError(f'{name} 必须为正整数')

    def plan(self, surface: SurfaceModel) -> PlanningResult:
        config = self.config
        outer = ShapelyPolygon(surface.scan_area.exterior).buffer(
            config['boundary_expansion_mm'], join_style=2)
        holes = [ShapelyPolygon(ring) for ring in surface.scan_area.interiors]
        area = outer
        for hole in holes:
            area = area.difference(hole)
        rectangles, _ = find_optimal_rectangles(
            area, surface.scan_area, config['rectangle_width_mm'],
            config['rectangle_height_mm'], config['scale_factor'],
            config['angle_step_deg'], config['max_cover_iterations'],
            config['candidate_grid_size_mm'])
        triangles = surface.triangles.copy()
        normals = surface.normals.copy()
        total_area = self._area(triangles)
        if total_area <= 0:
            raise ValueError('待测表面面积为空')
        viewpoints = []
        reason = 'iteration_limit'
        iterations = 0
        for iteration in range(config['max_refinement_cycles']):
            iterations = iteration + 1
            if not rectangles:
                reason = 'no_candidates'
                break
            new_views = self._map_viewpoints(rectangles, triangles, normals)
            if not new_views:
                reason = 'no_valid_viewpoints'
                break
            viewpoints.extend(new_views)
            remaining = self._remaining_mask(rectangles, new_views, triangles, normals)
            triangles, normals = triangles[remaining], normals[remaining]
            print(f'第 {iterations} 轮生成 {len(new_views)} 个视点，剩余 {len(triangles)} 个面片')
            if not len(triangles):
                reason = 'covered'
                break
            if iterations < config['max_refinement_cycles']:
                rectangles = generate_rectangles_for_clusters(
                    triangles[:, :, :2].tolist(), config['rectangle_width_mm'],
                    config['rectangle_height_mm'], config['scale_factor'],
                    config['angle_step_deg'], config['dbscan_eps_mm'],
                    config['dbscan_min_samples'])
        coverage = 1 - self._area(triangles) / total_area
        return PlanningResult(viewpoints, triangles, normals, reason, coverage, iterations)

    @staticmethod
    def _area(triangles) -> float:
        return float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                             triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2)

    def _map_viewpoints(self, rectangles: list[dict], triangles: np.ndarray, normals: np.ndarray) -> list[Viewpoint]:
        assignments = [[] for _ in rectangles]
        corners = [list(rect['rect'].exterior.coords)[:4] for rect in rectangles]
        # 重叠区域仍归属第一个完整包含它的矩形。
        for index, tri in enumerate(triangles):
            for rect_id, polygon in enumerate(corners):
                if all(point_in_polygon(p[0], p[1], polygon) for p in tri):
                    assignments[rect_id].append(index)
                    break
        viewpoints = []
        for rect_id, indices in enumerate(assignments):
            if len(indices) < self.config['triangle_threshold']:
                continue
            rectangle = rectangles[rect_id]
            plane = fit_plane(triangles[indices].reshape(-1, 3))
            if abs(plane[2]) < 1e-6:
                continue
            center = np.array(line_plane_intersection(plane, rectangle['center']))
            direction = -normalize_vector(normals[indices].mean(axis=0))
            position = center - self.config['viewpoint_offset_mm'] * direction
            if np.isfinite(position).all() and np.linalg.norm(direction) > 0:
                # 保留矩形索引直至覆盖检查，文件导出时才连续编号。
                viewpoints.append(Viewpoint(rect_id, rectangle['angle'], position, direction))
        return viewpoints

    def _remaining_mask(self, rectangles: list[dict], viewpoints: list[Viewpoint], triangles: np.ndarray, normals: np.ndarray) -> np.ndarray:
        remaining = np.ones(len(triangles), dtype=bool)
        cos_threshold = np.cos(np.radians(self.config['visibility_angle_threshold_deg']))
        distance = self.config['viewpoint_offset_mm']
        # 保留原有效深度范围，视场比例使用实际配置的工作距离。
        far_distance = 555.0
        for view in viewpoints:
            rectangle = rectangles[view.rectangle_id]
            center = np.asarray(rectangle['center'])
            corners = np.asarray(rectangle['rect'].exterior.coords)[:4]
            rays = np.column_stack(((corners - center) * far_distance / distance,
                                    np.full(4, far_distance)))
            rotation = rotation_matrix_from_vectors(np.array([0., 0., 1.]), view.direction)
            rays = rays @ rotation.T
            planes = np.cross(rays, np.roll(rays, -1, axis=0))
            planes /= np.linalg.norm(planes, axis=1)[:, None]
            planes[(planes @ view.direction) < 0] *= -1
            indices = np.flatnonzero(remaining)
            offsets = triangles[indices] - view.position
            height = offsets @ view.direction
            inside_depth = ((height >= 45) & (height <= far_distance)).all(axis=1)
            indices, offsets = indices[inside_depth], offsets[inside_depth]
            inside = (np.einsum('tvi,pi->tvp', offsets, planes) >= 0).all(axis=(1, 2))
            indices, offsets = indices[inside], offsets[inside]
            lengths = np.linalg.norm(offsets, axis=2)
            dot = np.einsum('tvi,ti->tv', offsets, normals[indices])
            angle_ok = ((lengths > 0) & (dot < cos_threshold * lengths)).all(axis=1)
            remaining[indices[angle_ok]] = False
        return remaining


def create_rotated_rectangle(center_x, center_y, width, height, angle_degrees) -> ShapelyPolygon:
    """在 XY 平面构造矩形，正角度从 +X 转向 +Y。"""
    angle_rad = math.radians(angle_degrees)
    half_width = width / 2
    half_height = height / 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    rotated_corners = []
    for dx, dy in corners:
        rotated_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        rotated_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        x = center_x + rotated_dx
        y = center_y + rotated_dy
        rotated_corners.append((x, y))

    return ShapelyPolygon(rotated_corners)


def normalize_vector(v) -> np.ndarray:
    """归一化向量"""
    vector = np.asarray(v, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def fit_plane(points) -> tuple[float, float, float, float]:
    """用最小奇异向量求平面法向，返回 (a, b, c, d)。"""
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('拟合平面至少需要三个有效点')
    center = points.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(points - center, full_matrices=False)
    if singular_values[1] < 1e-10:
        raise ValueError('共线点无法确定平面')
    normal = vectors[-1]
    return (*normal, -float(normal @ center))


def line_plane_intersection(plane, point_xy) -> tuple[float, float, float]:
    """求过指定 XY 点、平行局部 Z 轴的直线与平面的交点。"""
    a, b, c, d = plane
    x, y = point_xy
    if abs(c) < 1e-6:
        raise ValueError('拟合平面无法沿局部 Z 轴投影')
    return x, y, (-d - a * x - b * y) / c


def get_candidate_points(area, grid_size=5) -> list[tuple[float, float]]:
    """生成候选点：网格点和区域顶点"""
    candidates = set()

    min_x, min_y, max_x, max_y = area.bounds

    candidates.add((min_x, min_y))
    candidates.add((min_x, max_y))
    candidates.add((max_x, min_y))
    candidates.add((max_x, max_y))

    if isinstance(area, ShapelyPolygon):
        for point in area.exterior.coords:
            candidates.add((point[0], point[1]))
        for interior in area.interiors:
            for point in interior.coords:
                candidates.add((point[0], point[1]))

    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > 0 and y_range > 0:
        x_steps = max(1, int(x_range / grid_size))
        y_steps = max(1, int(y_range / grid_size))

        for i in range(x_steps + 1):
            x = min_x + i * x_range / x_steps
            for j in range(y_steps + 1):
                y = min_y + j * y_range / y_steps
                candidates.add((x, y))

    return list(candidates)


def find_optimal_rectangles(area, original_area, rect_width, rect_height, scale_factor,
                            angle_step, max_iterations, grid_size) -> tuple[list[dict], float]:
    """每轮选择新增覆盖面积最大的矩形，返回矩形列表和面积覆盖率。"""
    rectangles = []
    remaining_area = area
    min_area_threshold = 0.01

    scaled_width = rect_width * scale_factor
    scaled_height = rect_height * scale_factor

    print("=" * 50)
    print(f"开始矩形覆盖")
    print("=" * 50)

    candidate_points = get_candidate_points(original_area, grid_size=grid_size)
    print(f"基于原始边界生成 {len(candidate_points)} 个候选点")

    # 剩余区域只会缩小，因此上一轮的覆盖面积可作为本轮上界。
    candidates = []
    queue = []
    for center in candidate_points:
        for angle in range(0, 180, angle_step):
            rect = create_rotated_rectangle(*center, scaled_width, scaled_height, angle)
            if rect is None or not rect.is_valid:
                continue
            clipped = rect.intersection(area)
            if clipped.area > min_area_threshold:
                index = len(candidates)
                candidates.append((rect, clipped, center, angle))
                queue.append((-clipped.area, index, 1))
    heapq.heapify(queue)

    iteration = 0
    while queue and remaining_area.area > min_area_threshold and iteration < max_iterations:
        iteration += 1
        best_scaled_rect = None
        while queue:
            negative_area, index, evaluated_at = heapq.heappop(queue)
            rect, clipped, center, angle = candidates[index]
            if evaluated_at == iteration:
                best_scaled_rect, best_center, best_angle = rect, center, angle
                best_coverage = -negative_area
                break
            coverage = clipped.intersection(remaining_area).area
            if coverage > min_area_threshold:
                heapq.heappush(queue, (-coverage, index, iteration))

        if best_scaled_rect and best_coverage > min_area_threshold:
            original_rect = create_rotated_rectangle(best_center[0], best_center[1],
                                                     rect_width, rect_height, best_angle)
            rectangles.append({
                'rect': original_rect,
                'scaled_rect': best_scaled_rect,
                'angle': best_angle,
                'center': best_center,
                'coverage': best_coverage
            })

            remaining_area = remaining_area.difference(best_scaled_rect)
        else:
            print(f"迭代 {iteration}: 未找到合适的矩形")
            break

    if area.area > 0:
        covered_area = sum(r['coverage'] for r in rectangles)
        coverage_rate = (covered_area / area.area) * 100
    else:
        print("警告: 初始区域面积为0")
        coverage_rate = 0

    return rectangles, coverage_rate


def triangle_center(triangle) -> tuple[float, float]:
    """计算三角形的中心点(x,y)"""
    xs = [p[0] for p in triangle]
    ys = [p[1] for p in triangle]
    return (sum(xs) / 3.0, sum(ys) / 3.0)


def calculate_cluster_center(cluster_triangles) -> tuple[float, float]:
    """计算簇的中心点（基于所有顶点）"""
    all_points = []
    for tri in cluster_triangles:
        all_points.extend(tri)
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def find_best_angle_for_cluster(cluster_triangles, center_x, center_y, scaled_width, scaled_height, angle_step) -> tuple[float, int]:
    """为簇寻找最佳角度"""
    best_angle = 0
    best_coverage = 0

    for angle in range(0, 180, angle_step):

        scaled_rect = create_rotated_rectangle(center_x, center_y, scaled_width, scaled_height, angle)
        if not scaled_rect.is_valid:
            continue

        covered_count = 0
        for tri in cluster_triangles:

            if all(scaled_rect.contains(Point(point)) for point in tri):
                covered_count += 1

        if covered_count > best_coverage:
            best_coverage = covered_count
            best_angle = angle

    return best_angle, best_coverage


def generate_rectangles_for_clusters(triangles, rect_width, rect_height, scale_factor,
                                     angle_step, eps_factor, min_samples=1) -> list[dict]:
    """对三角形中心做 DBSCAN 聚类，每个有效簇生成一个矩形。"""
    scaled_width = rect_width * scale_factor
    scaled_height = rect_height * scale_factor

    eps = eps_factor

    print("=" * 50)
    print(f"开始矩形生成 ")
    print("=" * 50)

    if not triangles:
        print("没有三角形需要处理")
        return []

    centers = [triangle_center(tri) for tri in triangles]

    centers_arr = np.array(centers)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(centers_arr)

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            if -1 not in clusters:
                clusters[-1] = []
            clusters[-1].append(idx)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

    valid_clusters = {k: v for k, v in clusters.items() if k != -1}

    if not valid_clusters:
        print("没有形成有效的簇，无法生成矩形")
        return []

    rectangles = []
    start_time = time.time()

    for label, indices in valid_clusters.items():

        cluster_tris = [triangles[i] for i in indices]

        center_x, center_y = calculate_cluster_center(cluster_tris)

        best_angle, coverage = find_best_angle_for_cluster(
            cluster_tris, center_x, center_y, scaled_width, scaled_height, angle_step
        )

        original_rect = create_rotated_rectangle(center_x, center_y, rect_width, rect_height, best_angle)

        rectangles.append({
            'rect': original_rect,
            'scaled_rect': create_rotated_rectangle(center_x, center_y, scaled_width, scaled_height, best_angle),
            'angle': best_angle,
            'center': (center_x, center_y),
            'coverage': coverage,
            'cluster_size': len(cluster_tris)
        })

    elapsed_time = time.time() - start_time
    print(f"矩形生成完成，耗时: {elapsed_time:.2f}秒")
    print(f"总共生成矩形: {len(rectangles)}")

    return rectangles


def point_in_polygon(x, y, poly) -> bool:
    """判断点(x,y)是否在多边形poly内部"""
    n = len(poly)
    inside = False
    u1, v1 = poly[0]

    for i in range(1, n + 1):
        u2, v2 = poly[i % n]
        if y > min(v1, v2) and y <= max(v1, v2) and x <= max(u1, u2):
            if v1 != v2:
                x_intersect = (y - v1) * (u2 - u1) / (v2 - v1) + u1
            if v1 == v2 or x <= x_intersect:
                inside = not inside
        u1, v1 = u2, v2

    return inside


def rotation_matrix_from_vectors(v1, v2) -> np.ndarray:
    """计算从向量v1旋转到v2的旋转矩阵"""
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    cos_theta = np.dot(v1, v2)

    if abs(cos_theta + 1.0) < 1e-6:
        reference = np.eye(3)[np.argmin(np.abs(v1))]
        axis = normalize_vector(np.cross(v1, reference))
        return 2 * np.outer(axis, axis) - np.eye(3)
    elif abs(cos_theta - 1.0) < 1e-6:
        return np.eye(3)

    axis = np.cross(v1, v2)
    sin_theta = np.linalg.norm(axis)
    axis = normalize_vector(axis)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return R

