"""在局部 XY 平面沿长轴分组，用蚁群算法排序，再依次连接各组。

输出视点访问顺序，并用给定矩阵转换位置和方向。"""
from __future__ import annotations


import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict


def load_transform_matrix(matrix_file) -> np.ndarray:
    """从文件读取4x4变换矩阵（行优先），返回numpy数组"""
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"变换矩阵文件不存在: {matrix_file}")
    with open(matrix_file, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        row = list(map(float, line.strip().split()))
        if len(row) != 4:
            raise ValueError(f"矩阵行长度错误，应为4，实际{len(row)}")
        matrix.append(row)
    if len(matrix) != 4:
        raise ValueError(f"矩阵应有4行，实际{len(matrix)}")
    return np.array(matrix)

def transform_points_and_normals(points, normals, matrix) -> tuple[np.ndarray, np.ndarray]:
    """将点（齐次坐标）和法向量（仅旋转）从壁板坐标系转换到世界坐标系。"""
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    points_world = points @ R.T + t
    normals_world = normals @ R.T
    return points_world, normals_world


def euclidean_distance(p1, p2) -> float:
    """计算两点之间的欧氏距离（xy平面）"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def distance_matrix(points) -> np.ndarray:
    """计算所有点之间的距离矩阵（欧氏距离）"""
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean_distance(points[i], points[j])
            dist[i][j] = dist[j][i] = d
    return dist

def ant_colony_tsp(points, start_idx, end_idx, config) -> list[int]:
    """用蚁群算法求固定起终点的开放路径，返回点索引顺序。"""
    n = len(points)
    if n == 1:
        return [start_idx]
    if n == 2:
        return [start_idx, end_idx] if start_idx != end_idx else [start_idx]

    dist = distance_matrix(points)

    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                eta[i][j] = 1.0 / (dist[i][j] + 1e-10)

    tau = np.full((n, n), config['tau0'])

    best_path = None
    best_length = float('inf')

    for _ in range(config['max_iter']):
        all_paths = []
        all_lengths = []

        for _ in range(config['ant_count']):

            visited = [start_idx]
            current = start_idx
            unvisited = set(range(n)) - {start_idx}

            while unvisited:

                if len(unvisited) == 1:
                    next_city = unvisited.pop()
                    visited.append(next_city)
                    break
                else:

                    probs = []
                    total = 0.0
                    for j in unvisited:
                        # 指定终点留到最后访问。
                        if j == end_idx and len(unvisited) > 1:
                            prob = 0.0
                        else:
                            prob = (tau[current][j] ** config['alpha']) * (eta[current][j] ** config['beta'])
                        probs.append(prob)
                        total += prob
                    if total == 0:
                        # 概率全为零时，从非终点中随机选择。
                        candidates = list(unvisited)
                        if end_idx in candidates and len(candidates) > 1:
                            candidates.remove(end_idx)
                        next_city = random.choice(candidates)
                    else:
                        # 轮盘赌选择
                        probs = [p / total for p in probs]
                        cumsum = 0
                        r = random.random()
                        for idx, prob in enumerate(probs):
                            cumsum += prob
                            if r <= cumsum:
                                next_city = list(unvisited)[idx]
                                break
                    visited.append(next_city)
                    current = next_city
                    unvisited.remove(next_city)

            length = 0
            for i in range(len(visited)-1):
                length += dist[visited[i]][visited[i+1]]
            all_paths.append(visited)
            all_lengths.append(length)
            if length < best_length:
                best_length = length
                best_path = visited.copy()

        tau *= (1 - config['rho'])
        for path, length in zip(all_paths, all_lengths):
            delta = config['Q'] / length
            for i in range(len(path)-1):
                a, b = path[i], path[i+1]
                tau[a][b] += delta
                tau[b][a] += delta

    if best_path[-1] != end_idx:
        if end_idx in best_path:
            best_path.remove(end_idx)
            best_path.append(end_idx)
        else:
            best_path.append(end_idx)
    return best_path

def read_vp_txt(file_path) -> list[dict]:
    """读取视点编号、局部 XY 坐标和原始行，原始行供路径导出。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                print(f"Warning: Skipping invalid line: {line}")
                continue
            idx = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            points.append({
                'index': idx,
                'x': x,
                'y': y,
                'full_line': line
            })
    return points

def compute_direction_and_groups(points, step) -> tuple[list[list[int]], int, int, np.ndarray]:
    """用 PCA 求 XY 主轴并按 step 分组，返回分组、两端索引和方向。"""
    if len(points) < 2:
        raise ValueError("路径规划至少需要两个视点")
    if step <= 0:
        raise ValueError("路径分组间距必须大于 0")

    coords = np.asarray([[p['x'], p['y']] for p in points], dtype=np.float64)
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_max = np.argmax(eigvals)
    direction = np.asarray(eigvecs[:, idx_max], dtype=np.float64)
    proj = np.asarray(centered @ direction, dtype=np.float64)
    start_idx = np.argmin(proj)
    end_idx = np.argmax(proj)
    if proj[start_idx] > proj[end_idx]:
        direction = -direction
        proj = -proj
        start_idx, end_idx = end_idx, start_idx
    proj_min = float(proj[start_idx])
    proj_max = float(proj[end_idx])
    groups = defaultdict(list)
    for i, p in enumerate(points):
        t = float(proj[i])
        k = int((t - proj_min) // step)
        if k * step > proj_max - proj_min:
            k = int((proj_max - proj_min) // step)
        groups[k].append(i)
    group_keys = sorted(groups.keys())
    grouped_indices = [groups[k] for k in group_keys]
    return grouped_indices, start_idx, end_idx, direction

def process_groups(points, grouped_indices, start_idx, end_idx, algo_config) -> list[int]:
    """按长轴顺序连接各组，组内使用蚁群排序。

    后续组从距上一组终点最近的点出发，以组内距起点最远的点结束。
    """

    group_points = []
    for group in grouped_indices:
        group_points.append([(points[i]['x'], points[i]['y']) for i in group])

    overall_path = []

    first_group_indices = grouped_indices[0]
    first_group_coords = group_points[0]
    if start_idx not in first_group_indices:
        dists = [euclidean_distance((points[start_idx]['x'], points[start_idx]['y']), p) for p in first_group_coords]
        group_start = first_group_indices[np.argmin(dists)]
    else:
        group_start = start_idx
    start_coord = (points[group_start]['x'], points[group_start]['y'])
    max_dist = -1
    group_end = group_start
    for idx in first_group_indices:
        d = euclidean_distance(start_coord, (points[idx]['x'], points[idx]['y']))
        if d > max_dist:
            max_dist = d
            group_end = idx
    mapping = {orig: i for i, orig in enumerate(first_group_indices)}
    inv_mapping = {i: orig for orig, i in mapping.items()}
    start_local = mapping[group_start]
    end_local = mapping[group_end]
    local_path = ant_colony_tsp(first_group_coords, start_local, end_local, algo_config)
    first_path = [inv_mapping[i] for i in local_path]
    overall_path.extend(first_path)

    for g_idx in range(1, len(grouped_indices)):
        group_indices = grouped_indices[g_idx]
        group_coords = group_points[g_idx]
        prev_end = overall_path[-1]
        prev_end_coord = (points[prev_end]['x'], points[prev_end]['y'])
        dists = [euclidean_distance(prev_end_coord, p) for p in group_coords]
        group_start = group_indices[np.argmin(dists)]
        start_coord = (points[group_start]['x'], points[group_start]['y'])
        max_dist = -1
        group_end = group_start
        for idx in group_indices:
            d = euclidean_distance(start_coord, (points[idx]['x'], points[idx]['y']))
            if d > max_dist:
                max_dist = d
                group_end = idx
        mapping = {orig: i for i, orig in enumerate(group_indices)}
        inv_mapping = {i: orig for orig, i in mapping.items()}
        start_local = mapping[group_start]
        end_local = mapping[group_end]
        local_path = ant_colony_tsp(group_coords, start_local, end_local, algo_config)
        group_path = [inv_mapping[i] for i in local_path]
        overall_path.extend(group_path)

    return overall_path

def save_path_txt(overall_path, points, output_path) -> None:
    """按访问顺序写出视点原始数据行。"""
    with open(output_path, 'w') as f:
        for idx in overall_path:
            f.write(points[idx]['full_line'] + '\n')

def visualize_path(overall_path, points, vis_config, png_path) -> None:
    """可视化路径，保存为PNG"""
    route_points = []
    route_indices = []
    for idx in overall_path:
        route_points.append([points[idx]['x'], points[idx]['y']])
        route_indices.append(points[idx]['index'])
    route_points = np.array(route_points)

    plt.figure(figsize=vis_config['figsize'])
    plt.plot(route_points[:, 0], route_points[:, 1], 'r-', linewidth=vis_config['linewidth'], alpha=0.8, label='Path')
    all_x = [p['x'] for p in points]
    all_y = [p['y'] for p in points]
    plt.scatter(all_x, all_y, c='blue', s=vis_config['point_size'], alpha=0.8, label='View Points')
    plt.scatter(route_points[0, 0], route_points[0, 1], c='green', s=200, marker=vis_config['start_marker'], label='Start')
    plt.scatter(route_points[-1, 0], route_points[-1, 1], c='red', s=200, marker=vis_config['end_marker'], label='End')

    plt.xlabel('X (mm)', fontsize=12)
    plt.ylabel('Y (mm)', fontsize=12)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=vis_config['dpi'], bbox_inches='tight')
    print(f"Visualization saved to {png_path}")

def plan_path(config, viewpoint_file, matrix, output_dir) -> bool:
    """生成分组访问顺序，保存局部路径、预览和坐标转换后的路径。"""
    print("开始路径规划")
    input_file = viewpoint_file
    output_dir = os.fspath(output_dir)
    output_txt = os.path.join(output_dir, 'Path_base_pannel.txt')
    output_png = os.path.join(output_dir, 'Path_visualization.png')
    world_path_txt = os.path.join(output_dir, 'Path_base_world.txt')
    group_step = config['group_step_mm']
    algorithm_config = config['ant_colony']
    ant_count = algorithm_config['ant_count']
    max_iter = algorithm_config['max_iterations']
    alpha = algorithm_config['alpha']
    beta = algorithm_config['beta']
    rho = algorithm_config['evaporation_rate']
    Q = algorithm_config['pheromone_deposit']
    tau0 = algorithm_config['initial_pheromone']

    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return False

    print(f"读取视点文件: {input_file}")
    points = read_vp_txt(input_file)
    if not points:
        print("错误：没有有效的视点数据")
        return False
    print(f"共加载 {len(points)} 个视点")

    print("计算主方向并分组...")
    if len(points) == 1:
        grouped_indices, start_idx, end_idx, direction = [[0]], 0, 0, np.array([1., 0.])
    else:
        grouped_indices, start_idx, end_idx, direction = compute_direction_and_groups(points, group_step)
    print(f"分组数: {len(grouped_indices)}")
    print(f"主方向: {direction}")

    algo_config = {
        'ant_count': ant_count,
        'max_iter': max_iter,
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'Q': Q,
        'tau0': tau0
    }

    print("蚁群算法规划路径...")
    overall_path = process_groups(points, grouped_indices, start_idx, end_idx, algo_config)
    print(f"路径包含 {len(overall_path)} 个视点")

    save_path_txt(overall_path, points, output_txt)
    print(f"路径文件已保存: {output_txt}")

    visual_config = config['visualization']
    visualize_path(
        overall_path,
        points,
        {
            'figsize': visual_config['figure_size'],
            'dpi': visual_config['dpi'],
            'point_size': visual_config['point_size'],
            'start_marker': visual_config['start_marker'],
            'end_marker': visual_config['end_marker'],
            'linewidth': visual_config['line_width'],
        },
        output_png,
    )

    print("转换到世界坐标系...")
    try:
        panel_points = []
        panel_normals = []
        with open(output_txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            x, y, z = map(float, parts[2:5])
            nx, ny, nz = map(float, parts[5:8])
            panel_points.append([x, y, z])
            panel_normals.append([nx, ny, nz])
        panel_points = np.array(panel_points)
        panel_normals = np.array(panel_normals)

        world_points, world_normals = transform_points_and_normals(panel_points, panel_normals, matrix)

        with open(world_path_txt, 'w') as f:
            for i, (line, pt, norm) in enumerate(zip(lines, world_points, world_normals)):
                parts = line.strip().split()

                parts[2:5] = [f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"]
                parts[5:8] = [f"{norm[0]:.6f}", f"{norm[1]:.6f}", f"{norm[2]:.6f}"]
                f.write(" ".join(parts) + "\n")
        print(f"世界坐标系路径已保存: {world_path_txt}")
    except Exception as e:
        print(f"转换到世界坐标系失败: {e}")
        return False

    return True

