"""前处理和规划阶段共享的数据。"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon


@dataclass
class SurfaceModel:
    """局部 XY 为板面、Z 为厚度；矩阵将局部坐标变换回 STEP 坐标。"""
    triangles: np.ndarray
    normals: np.ndarray
    boundary: Polygon
    scan_area: Polygon
    local_to_world: np.ndarray
    metadata: dict = field(default_factory=dict)
    cad_shape: object = None
    preview_faces: list = field(default_factory=list)


@dataclass
class Viewpoint:
    rectangle_id: int
    angle: float
    position: np.ndarray
    direction: np.ndarray


@dataclass
class PlanningResult:
    viewpoints: list[Viewpoint]
    remaining_triangles: np.ndarray
    remaining_normals: np.ndarray
    stop_reason: str
    coverage_rate: float
    iterations: int
