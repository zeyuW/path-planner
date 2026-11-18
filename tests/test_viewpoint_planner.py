"""覆盖循环、矩形索引和几何计算回归。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import box

from vp_planner import SurfaceModel, ViewpointPlanner
from vp_planner.file_io import export_planning_result
from vp_planner.planner import fit_plane, rotation_matrix_from_vectors, line_plane_intersection


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.triangles = np.array([[[0, 0, 0], [0, 0, 10], [0, 10, 0]],
                                   [[0, 10, 0], [0, 0, 10], [0, 10, 10]]], dtype=float)
        self.triangles = self.triangles[:, :, [1, 2, 0]]
        self.normals = np.array([[0, 0, -1.], [0, 0, -1.]])
        self.config = dict(rectangle_width_mm=30, rectangle_height_mm=30,
                           scale_factor=.9, angle_step_deg=45, max_cover_iterations=10,
                           candidate_grid_size_mm=10, triangle_threshold=1,
                           viewpoint_offset_mm=300, visibility_angle_threshold_deg=120,
                           dbscan_eps_mm=10, dbscan_min_samples=1,
                           max_refinement_cycles=2, boundary_expansion_mm=0)

    def test_flat_surface_and_export(self):
        surface = SurfaceModel(self.triangles, self.normals, box(0, 0, 10, 10),
                               box(0, 0, 10, 10), np.eye(4))
        result = ViewpointPlanner(self.config).plan(surface)
        self.assertEqual(result.stop_reason, 'covered')
        self.assertEqual(result.coverage_rate, 1)
        self.assertTrue(result.viewpoints)
        self.assertEqual(len(result.remaining_triangles), 0)
        for view in result.viewpoints:
            self.assertAlmostEqual(view.position[2], -300)
        parent = Path(__file__).resolve().parents[1] / '.tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            export_planning_result(result, temp)
            data = np.loadtxt(Path(temp) / 'VP.txt', ndmin=2)
            self.assertEqual(data.shape, (len(result.viewpoints), 8))
            first_column = (Path(temp) / 'VP.txt').read_text().split()[0]
            self.assertEqual(int(first_column), 1)
            self.assertTrue((Path(temp) / 'planning_report.json').is_file())

    def test_filtered_rectangle_keeps_original_index(self):
        rectangles = [{'rect': box(100, 100, 130, 130), 'center': (115, 115), 'angle': 0},
                      {'rect': box(-5, -5, 15, 15), 'center': (5, 5), 'angle': 0}]
        planner = ViewpointPlanner(self.config)
        views = planner._map_viewpoints(rectangles, self.triangles, self.normals)
        self.assertEqual([view.rectangle_id for view in views], [1])
        mask = planner._remaining_mask(rectangles, views, self.triangles, self.normals)
        self.assertFalse(mask.any())

    def test_no_valid_viewpoints_is_reported(self):
        surface = SurfaceModel(self.triangles, self.normals, box(0, 0, 10, 10),
                               box(0, 0, 10, 10), np.eye(4))
        self.config['triangle_threshold'] = 100
        result = ViewpointPlanner(self.config).plan(surface)
        self.assertEqual(result.stop_reason, 'no_valid_viewpoints')
        self.assertEqual(result.coverage_rate, 0)
        self.assertEqual(len(result.remaining_triangles), 2)

    def test_plane_and_rotations(self):
        plane = fit_plane(self.triangles.reshape(-1, 3))
        self.assertAlmostEqual(abs(plane[2]), 1)
        for direction in [np.array([.8, .6, 0]), np.array([-1., 0, 0])]:
            rotation = rotation_matrix_from_vectors(np.array([1., 0, 0]), direction)
            np.testing.assert_allclose(rotation @ [1., 0, 0], direction, atol=1e-12)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_tilted_surface_uses_xy_and_preserves_depth(self):
        triangles = self.triangles.copy()
        triangles[:, :, 2] = .1 * triangles[:, :, 0] + .2 * triangles[:, :, 1] + 7
        normal = np.array([.1, .2, -1.])
        normal /= np.linalg.norm(normal)
        normals = np.tile(normal, (2, 1))
        plane = fit_plane(triangles.reshape(-1, 3))
        np.testing.assert_allclose(line_plane_intersection(plane, (3, 6)), (3, 6, 8.5))
        rectangle = {'rect': box(-5, -5, 15, 15), 'center': (3, 6), 'angle': 0}
        planner = ViewpointPlanner(self.config)
        views = planner._map_viewpoints([rectangle], triangles, normals)
        self.assertEqual(len(views), 1)
        np.testing.assert_allclose(views[0].position + 300 * views[0].direction, (3, 6, 8.5))
        self.assertFalse(planner._remaining_mask([rectangle], views, triangles, normals).any())

    def test_path_reader_projects_xy_not_depth(self):
        from path_planner.planner import read_vp_txt, compute_direction_and_groups
        parent = Path(__file__).resolve().parents[1] / '.tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            path = Path(temp) / 'VP.txt'
            path.write_text('1 0 10 2 -300 0 0 1\n2 0 110 2 -900 0 0 1\n')
            points = read_vp_txt(path)
            self.assertEqual([(p['x'], p['y']) for p in points], [(10, 2), (110, 2)])
            _, _, _, direction = compute_direction_and_groups(points, 50)
            self.assertAlmostEqual(abs(direction[0]), 1)


if __name__ == '__main__':
    unittest.main()
