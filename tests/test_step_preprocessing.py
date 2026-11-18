"""选面回归：背面、断开的面、曲面类型、孔洞与可逆坐标。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vp_planner.file_io import parse_area_file, filter_surface_by_area
from vp_planner.file_io import prepare_step_model
from vp_planner.surface import (
    FaceMesh, _triangle_geometry, _select_patch, _local_frame,
    _refine_triangles, _project_boundary,
    _select_automatic_patch, _mesh_faces,
)


def face(fid, x, neighbors=(), reverse=False):
    tri = np.array([[[x, 0, 0], [x+2, 0, 0], [x, 0, 2]]], dtype=float)
    if reverse:
        tri = tri[:, [0, 2, 1]]
    normals, areas = _triangle_geometry(tri)
    return FaceMesh(fid, None, 'BSplineSurface', tri, normals, areas, set(neighbors))


class GeometryTests(unittest.TestCase):
    def test_grow_shared_faces_without_back_or_disconnected_face(self):
        faces = [face(1, 0, [2, 3]), face(2, 2, [1]),
                 face(3, 0, [1], reverse=True), face(4, 100)]
        selected, seed, _ = _select_patch(faces, [0, -1, 0], 25, 15)
        self.assertEqual(seed, 1)
        self.assertEqual([f.face_id for f in selected], [1, 2])
        selected, _, _ = _select_patch(faces, [0, -1, 0], 25, 15, 4)
        self.assertEqual([f.face_id for f in selected], [4])
        with self.assertRaises(ValueError):
            _select_patch(faces, [0, -1, 0], 25, 15, 3)

    def test_frame_and_refinement_preserve_geometry(self):
        tri = face(1, 5).triangles
        original_normals, original_area = _triangle_geometry(tri)
        refined = _refine_triangles(tri, 0.3)
        normals, areas = _triangle_geometry(refined)
        self.assertAlmostEqual(areas.sum(), original_area.sum())
        np.testing.assert_allclose(normals, np.tile(original_normals, (len(normals), 1)))
        for i in range(3):
            self.assertLessEqual(np.linalg.norm(refined[:, i]-refined[:, (i+1) % 3], axis=1).max(), 0.3+1e-10)
        origin, axes = _local_frame(tri)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(axes), 1)
        np.testing.assert_allclose(((tri-origin) @ axes) @ axes.T+origin, tri, atol=1e-12)

    def test_projection_rejects_disconnected_or_overlapping_sheets(self):
        tri = np.array([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], dtype=float)
        with self.assertRaises(ValueError):
            _project_boundary(np.concatenate((tri, tri+[0, 20, 0])))
        with self.assertRaises(ValueError):
            _project_boundary(np.concatenate((tri, tri+[0, 0, 2])))


class StepIntegrationTests(unittest.TestCase):
    def test_perforated_solid_exports_only_one_side_and_keeps_hole(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        box = BRepPrimAPI_MakeBox(10, 2, 8).Shape()
        drill = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5, -1, 4), gp_Dir(0, 1, 0)), 1, 4).Shape()
        shape = BRepAlgoAPI_Cut(box, drill).Shape()
        workspace = Path(__file__).resolve().parents[1]
        temp_parent = workspace / '.tmp'
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as temp:
            root = Path(temp)
            assert root.resolve().is_relative_to(workspace)
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            writer.Write(str(root/'input.step'))
            prepare_step_model(root/'input.step', root/'surface.txt', root/'area.txt',
                               expansion_mm=2, mesh_size_mm=1,
                               target_direction=[0, -1, 0],
                               linear_deflection_mm=0.01, extraction_dir=root/'extraction')
            report = json.loads((root/'extraction/selection.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['selected_face_ids']), 1)
            self.assertEqual(report['hole_count'], 1)
            area, outer, holes = parse_area_file(root/'area.txt')
            original, _, original_holes = parse_area_file(root/'extraction/surface_boundary.txt')
            self.assertTrue(area.covers(original))
            self.assertAlmostEqual(holes.area, original_holes.area, places=6)
            self.assertAlmostEqual(original.area, 80-np.pi, delta=0.03)
            data = np.loadtxt(root/'surface.txt')
            local = data[:, 1:10].reshape(-1, 3)
            transform = np.array(report['local_to_world'])
            world = local @ transform[:3, :3].T + transform[:3, 3]
            np.testing.assert_allclose(world[:, 1], 0, atol=1e-7)
            self.assertTrue(np.all(data[:, 12] < -0.99))
            filter_surface_by_area(root/'area.txt', root/'surface.txt', root/'remain.txt')
            self.assertEqual(len(np.loadtxt(root/'remain.txt')), len(data))
            self.assertTrue((root/'extraction/target_surface.step').is_file())

    def test_automatic_selection_and_frame_follow_rigid_transform(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        faces = _mesh_faces(BRepPrimAPI_MakeBox(100, 2, 18).Shape(), 0.05, 10)
        selected, _, _, info = _select_automatic_patch(faces, 25, 15)
        self.assertTrue(info['side_ambiguous'])
        self.assertAlmostEqual(sum(face.area for face in selected), 1800)
        triangles = np.concatenate([face.triangles for face in selected])
        origin, axes = _local_frame(triangles)
        baseline = (triangles - origin) @ axes
        rng = np.random.default_rng(128)
        for _ in range(12):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= np.linalg.det(rotation)
            translation = rng.normal(size=3) * 1000
            moved = []
            for face in faces:
                moved.append(FaceMesh(face.face_id, None, face.kind,
                                      face.triangles @ rotation.T + translation,
                                      face.normals @ rotation.T, face.areas,
                                      face.neighbors.copy()))
            result, _, _, _ = _select_automatic_patch(moved, 25, 15)
            self.assertEqual([f.face_id for f in result], [f.face_id for f in selected])
            moved_triangles = np.concatenate([f.triangles for f in result])
            moved_origin, moved_axes = _local_frame(moved_triangles)
            np.testing.assert_allclose((moved_triangles - moved_origin) @ moved_axes,
                                       baseline, atol=1e-8)

    def test_normal_entry_point_uses_automatic_selection(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        transform = gp_Trsf()
        transform.SetRotation(gp_Ax1(gp_Pnt(30, 20, 10), gp_Dir(1, 2, 3)), 1.23)
        shape = BRepBuilderAPI_Transform(BRepPrimAPI_MakeBox(100, 2, 18).Shape(), transform, True).Shape()
        parent = Path(__file__).resolve().parents[1] / '.tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            writer.Write(str(root / 'rotated.step'))
            prepare_step_model(root / 'rotated.step', root / 'surface.txt', root / 'area.txt',
                               expansion_mm=2, mesh_size_mm=10)
            report = json.loads((root / 'extraction/selection.json').read_text(encoding='utf-8'))
            self.assertIsNone(report['target_direction'])
            self.assertAlmostEqual(report['surface_area_mm2'], 1800, places=6)
            self.assertTrue(report['automatic_selection']['side_ambiguous'])


if __name__ == '__main__':
    unittest.main()
