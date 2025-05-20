import copy
import json
import os
import time
from typing import List

import numpy as np
import open3d
import pylfda
import trimesh
import yaml

from InlayAdaptation import InlayAdaptation
from inlaylast import inlayPostWarp
from register import MeshRegistration
from utils import find_changed_faces, find_new_points


def remove_duplicate(
    mesh: open3d.geometry.TriangleMesh,
) -> open3d.geometry.TriangleMesh:
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


class InlayGeneration:
    def __init__(
        self,
        tid: int,
        prep_tooth: open3d.geometry.TriangleMesh,
        inlay_inner: open3d.geometry.TriangleMesh,
        upper_scan: open3d.geometry.TriangleMesh,
        lower_scan: open3d.geometry.TriangleMesh,
        adjacent_teeth: List[open3d.geometry.TriangleMesh],
        standard: open3d.geometry.TriangleMesh,
        save_path: str = None,
        paras: dict = None,
    ) -> None:
        self.configs = None
        self.tooth_lib_configs = None
        self.tid = tid
        if tid:
            if tid < 30:
                self.prep_scan, self.anta_scan = upper_scan, lower_scan
            else:
                self.prep_scan, self.anta_scan = lower_scan, upper_scan
        self.get_configs()
        self.paras = paras
        self.load_paras()
        self.inlay_inner = inlay_inner
        self.prep_tooth = prep_tooth
        self.adjacent_teeth = adjacent_teeth
        self.lib_tooth = standard
        if self.configs["isSave"] and save_path is None:
            self.save_path = self.configs["savePath"]
        else:
            self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        self.configs["teeth_num"] = str(self.tid)
        self.configs["save_path"] = self.save_path
        self.configs["tooth_lib_configs"] = self.tooth_lib_configs

    def get_configs(self):
        with open("configs.yaml", "r") as f:
            self.configs = yaml.safe_load(f)
        if self.tid:
            with open("config/st_tooth_remesh/config.json", "r") as f:
                self.tooth_lib_configs = json.load(f)[str(self.tid)]

    def load_paras(self):
        if self.paras:
            for key, value in self.paras.items():
                if value is not None:
                    self.configs[key] = value

    def tri2o3d(self, mesh_trimesh):
        import copy

        vertices = np.asarray(mesh_trimesh.vertices)
        faces = np.asarray(mesh_trimesh.faces)
        vertex_normals = copy.deepcopy(np.asarray(mesh_trimesh.vertex_normals))
        mesh_open3d = open3d.geometry.TriangleMesh()
        mesh_open3d.vertices = open3d.utility.Vector3dVector(vertices)
        mesh_open3d.triangles = open3d.utility.Vector3iVector(faces)
        mesh_open3d.vertex_normals = open3d.utility.Vector3dVector(vertex_normals)
        return mesh_open3d

    def o3d2tri(self, mesh_open3d):
        vertices = np.asarray(mesh_open3d.vertices)
        faces = np.asarray(mesh_open3d.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def stitch(self):
        # assign data
        inlay_inner = pylfda.Mesh()
        if isinstance(self.inner_dilation, trimesh.Trimesh):
            dilation = self.inner_dilation.as_open3d
        else:
            dilation = self.inner_dilation
        inlay_inner.vertices = np.asarray(dilation.vertices)
        inlay_inner.faces = np.asarray(dilation.triangles)

        inlay_outer = pylfda.Mesh()
        if isinstance(self.inlay_outer, trimesh.Trimesh):
            outer = self.tri2o3d(self.inlay_outer)
        else:
            outer = self.inlay_outer
        inlay_outer.vertices = np.asarray(outer.vertices)
        inlay_outer.faces = np.asarray(outer.triangles)
        result_mesh = pylfda.Mesh()

        max_stitching_distance = 1
        success = pylfda.stitch(
            inlay_inner, inlay_outer, result_mesh, max_stitching_distance
        )
        if success:
            print("The gap was filled fully")
        else:
            print("The gap was filled partially")
        self.stitched_inlay = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(result_mesh.vertices),
            open3d.utility.Vector3iVector(result_mesh.faces),
        )
        self.stitched_inlay.compute_vertex_normals()

    def remove_overlaps(
        self, A: open3d.geometry.TriangleMesh, B: open3d.geometry.TriangleMesh
    ) -> None:
        """remove all vertices of A that are also in B

        Args:
            A (open3d.geometry.TriangleMesh): _description_
            B (open3d.geometry.TriangleMesh): _description_
        """
        A_ = copy.deepcopy(A)
        verts_A = np.asarray(A.vertices)
        verts_B = np.asarray(B.vertices)
        vert_overlap_ids = np.where(np.all(verts_A == verts_B, axis=0))[0]
        A_.remove_vertices_by_index(vert_overlap_ids)
        return A_

    def run(self) -> None:
        # self.register_lib_tooth()

        # registeration =============================================
        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(self.configs["save_path"], f"prep_scan_{self.tid}.ply"),
                self.prep_scan,
            )
            open3d.io.write_triangle_mesh(
                os.path.join(self.configs["save_path"], f"anta_scan_{self.tid}.ply"),
                self.anta_scan,
            )
        mesh_registration = MeshRegistration(self.configs)
        self.inlay_outer, self.inner_dilation = mesh_registration.run(
            self.lib_tooth, self.prep_tooth, self.inlay_inner, self.anta_scan
        )

        s = time.time()
        inlay_adaptation = InlayAdaptation()
        inlay_adaptation.setConfig(self.configs)
        self.inlay_outer = self.tri2o3d(self.inlay_outer)
        # inlay_outer_vertex_normals = self.inlay_outer.vertex_normals

        # # occlusal adaptation =======================================
        if self.configs["adjust_crown"]:
            self.inlay_outer = inlay_adaptation.calcuCrash(
                self.inlay_outer, self.anta_scan
            )  # occlusal
        print("8. occlusal adaptation time: ", time.time() - s)
        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"10_occlusal_inlay_outer_{self.tid}.ply"
                ),
                self.inlay_outer,
            )
        # proximal adaptation =======================================
        s = time.time()
        if self.configs["adjust_crown"]:
            for adjacent_tooth in self.adjacent_teeth:
                t = time.time()
                prep_scan_without_one_adj = self.remove_overlaps(
                    self.prep_scan, adjacent_tooth
                )
                print(f"remove_overlaps took {time.time() - t}s")
                self.inlay_outer = inlay_adaptation.calcuCrash(
                    self.inlay_outer, prep_scan_without_one_adj
                )  # proximal
        print("9. proximal adaptation time: ", time.time() - s)
        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"11_proximal_inlay_outer_{self.tid}.ply"
                ),
                self.inlay_outer,
            )

        # fill gap ==================================================
        s = time.time()
        self.stitch()
        print("10. stitching time: ", time.time() - s)

        if isinstance(self.inlay_outer, trimesh.Trimesh):
            self.inlay_outer = self.inlay_outer.as_open3d
        self.inlay_outer.compute_vertex_normals()
        if isinstance(self.stitched_inlay, trimesh.Trimesh):
            self.stitched_inlay = self.stitched_inlay.as_open3d
        self.stitched_inlay.compute_vertex_normals()

        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"12_final_inlay_outer_{self.tid}.ply"
                ),
                self.inlay_outer,
            )
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"13_stitched_inlay_{self.tid}.ply"
                ),
                self.stitched_inlay,
            )
        if self.configs["adjust_crown"]:
            surface_verts = np.array(self.inlay_outer.vertices).astype(np.float64)
            surface_faces = np.array(self.inlay_outer.triangles).astype(np.int32)
            inlay_verts = np.array(self.stitched_inlay.vertices).astype(np.float64)
            inlay_faces = np.array(self.stitched_inlay.triangles).astype(np.int32)
            self.inlay_outer = inlayPostWarp(surface_verts, surface_faces, inlay_verts, inlay_faces)
            
            self.stitch()
            if self.configs["isSave"]:
                open3d.io.write_triangle_mesh(
                    os.path.join(
                        self.configs["save_path"],
                        f"15_stitch_fixed_inlay_outer_{self.tid}.ply",
                    ),
                    self.tri2o3d(self.inlay_outer),
                )
                open3d.io.write_triangle_mesh(
                    os.path.join(
                        self.configs["save_path"],
                        f"14_stitch_fixed_inlay_{self.tid}.ply",
                    ),
                    self.stitched_inlay,
                )

    def run_occlu(self) -> None:
        # registeration =============================================
        self.inner_dilation = self.inlay_inner  # for occlusal adaptation
        self.inlay_outer = self.lib_tooth  # for occlusal adaptation
        s = time.time()
        inlay_adaptation = InlayAdaptation()
        inlay_adaptation.setConfig(self.configs)
        mesh_registration = MeshRegistration(self.configs)
        boundarySet, _ = mesh_registration.getBoundaryPoints(self.inner_dilation)
        self.inlay_outer = mesh_registration.doSubMeshTps(
            boundarySet, self.inlay_outer, mesh_registration.lastSubmehs_tps_dis
        )
        
        if isinstance(self.inlay_outer, trimesh.Trimesh):
            self.inlay_outer = self.tri2o3d(self.inlay_outer)
        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"4_boundary_TPSed_{self.tid}.ply"
                ),
                self.inlay_outer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ,
            )
        # fill gap ==================================================
        s = time.time()
        self.stitch()
        print("10. stitching time: ", time.time() - s)

        if isinstance(self.inlay_outer, trimesh.Trimesh):
            self.inlay_outer = self.inlay_outer.as_open3d
        self.inlay_outer.compute_vertex_normals()
        if isinstance(self.stitched_inlay, trimesh.Trimesh):
            self.stitched_inlay = self.stitched_inlay.as_open3d
        self.stitched_inlay.compute_vertex_normals()

        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"12_final_inlay_outer_{self.tid}.ply"
                ),
                self.inlay_outer,
            )
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"], f"13_stitched_inlay_{self.tid}.ply"
                ),
                self.stitched_inlay,
            )

        surface_verts = np.array(self.inlay_outer.vertices).astype(np.float64)
        surface_faces = np.array(self.inlay_outer.triangles).astype(np.int32)
        inlay_verts = np.array(self.stitched_inlay.vertices).astype(np.float64)
        inlay_faces = np.array(self.stitched_inlay.triangles).astype(np.int32)
        self.inlay_outer = inlayPostWarp(surface_verts, surface_faces, inlay_verts, inlay_faces)
            
        self.stitch()
        if self.configs["isSave"]:
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"],
                    f"15_stitch_fixed_inlay_outer_{self.tid}.ply",
                ),
                self.tri2o3d(self.inlay_outer),
            )
            open3d.io.write_triangle_mesh(
                os.path.join(
                    self.configs["save_path"],
                    f"14_stitch_fixed_inlay_{self.tid}.ply",
                ),
                self.stitched_inlay,
            )

    def get_inlay_outer(self) -> open3d.geometry.TriangleMesh:
        return self.inlay_outer

    def get_stitched_inlay(self) -> open3d.geometry.TriangleMesh:
        return self.stitched_inlay

    def get_inlay_inner(self) -> open3d.geometry.TriangleMesh:
        return self.inner_dilation
