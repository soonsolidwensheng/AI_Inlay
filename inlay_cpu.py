import os
import yaml
import json
import time
import copy
import open3d
import trimesh
import numpy as np

import pylfda
from typing import List
from register import MeshRegistration
from InlayAdaptation import InlayAdaptation
from utils import read_mesh



def remove_duplicate(mesh: open3d.geometry.TriangleMesh) -> open3d.geometry.TriangleMesh:
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


class InlayGeneration():
    def __init__(self, 
                 tid: int,
                 prep_tooth: open3d.geometry.TriangleMesh,
                 inlay_inner: open3d.geometry.TriangleMesh,
                 upper_scan: open3d.geometry.TriangleMesh,
                 lower_scan: open3d.geometry.TriangleMesh,
                 adjacent_teeth: List[open3d.geometry.TriangleMesh],
                 standard: open3d.geometry.TriangleMesh,
                 save_path: str=None) -> None:
        self.configs = None
        self.tooth_lib_configs = None
        self.tid = tid
        self.prep_tooth = prep_tooth
        self.inlay_inner = inlay_inner
        if tid < 30:
            self.prep_scan, self.anta_scan = upper_scan, lower_scan  
        else:
            self.prep_scan, self.anta_scan = lower_scan, upper_scan
        self.adjacent_teeth = adjacent_teeth
        self.lib_tooth = standard
        self.get_configs()
        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def get_configs(self):
        with open('configs.yaml', 'r') as f:
            self.configs = yaml.safe_load(f)
        with open('config/st_tooth_remesh/config.json', 'r') as f:
            self.tooth_lib_configs = json.load(f)[str(self.tid)]

    def register_lib_tooth(self):
        self.lib_tooth = read_mesh(os.path.join('config/st_tooth_remesh', f'{self.tid}.ply'))
        prep_bbox = self.prep_tooth.get_axis_aligned_bounding_box()
        lib_tooth_bbox = self.lib_tooth.get_axis_aligned_bounding_box()
        # scale lib tooth
        occlu_surf_size_ratio = (prep_bbox.get_extent()[0] * prep_bbox.get_extent()[2]) \
            / (lib_tooth_bbox.get_extent()[0] * lib_tooth_bbox.get_extent()[2])
        scale_factor = np.sqrt(occlu_surf_size_ratio)
        self.lib_tooth.scale(scale_factor, self.lib_tooth.get_center())
        # translate lib tooth
        self.lib_tooth.translate(prep_bbox.get_center() - self.lib_tooth.get_center())
        
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
    

    def stitch(self):
        # assign data
        inlay_inner = pylfda.Mesh()
        if isinstance(self.inner_dilation, trimesh.Trimesh):
            dilation = self.inner_dilation.as_open3d
        inlay_inner.vertices = np.asarray(dilation.vertices)
        inlay_inner.faces = np.asarray(dilation.triangles)

        inlay_outer = pylfda.Mesh()
        if isinstance(self.inner_dilation, trimesh.Trimesh):
            outer = self.inlay_outer
        inlay_outer.vertices = np.asarray(outer.vertices)
        inlay_outer.faces = np.asarray(outer.triangles)
        result_mesh = pylfda.Mesh()

        max_stitching_distance = 1
        success = pylfda.stitch(inlay_inner, inlay_outer, result_mesh, max_stitching_distance)
        if success:
            print('The gap was filled fully')  
        else :  
            print('The gap was filled partially')
        self.stitched_inlay = open3d.geometry.TriangleMesh(open3d.utility.Vector3dVector(result_mesh.vertices), 
                                                           open3d.utility.Vector3iVector(result_mesh.faces))
        self.stitched_inlay.compute_vertex_normals()

    def remove_overlaps(self,
                        A: open3d.geometry.TriangleMesh, 
                        B: open3d.geometry.TriangleMesh
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
        self.configs['teeth_num'] = str(self.tid)
        self.configs['save_path'] = self.save_path
        self.configs['tooth_lib_configs'] = self.tooth_lib_configs
        mesh_registration = MeshRegistration(self.configs)
        self.inlay_outer, self.inner_dilation = mesh_registration.run(self.lib_tooth, self.prep_tooth, self.inlay_inner)
        
        
        s = time.time()
        inlay_adaptation = InlayAdaptation()
        inlay_adaptation.setConfig(self.configs)
        self.inlay_outer = self.tri2o3d(self.inlay_outer)
        # inlay_outer_vertex_normals = self.inlay_outer.vertex_normals

        # # occlusal adaptation =======================================
        # self.inlay_outer = inlay_adaptation.calcuCrash(self.inlay_outer, self.anta_scan) # occlusal
        # print("8. occlusal adaptation time: ", time.time()-s)
        # if self.configs['isSave']:
        #     open3d.io.write_triangle_mesh(os.path.join(self.configs['save_path'], f'10_occlusal_inlay_outer_{self.tid}.ply'), self.inlay_outer)
        # proximal adaptation =======================================
        s = time.time()
        for adjacent_tooth in self.adjacent_teeth:
            t = time.time()
            prep_scan_without_one_adj = self.remove_overlaps(self.prep_scan, adjacent_tooth)
            print(f'remove_overlaps took {time.time()-t}s')
            self.inlay_outer = inlay_adaptation.calcuCrash(self.inlay_outer, prep_scan_without_one_adj) # proximal
        print("9. proximal adaptation time: ", time.time()-s)
        if self.configs['isSave']:
            open3d.io.write_triangle_mesh(os.path.join(self.configs['save_path'], f'11_proximal_inlay_outer_{self.tid}.ply'), self.inlay_outer)
        
        # fill gap ==================================================
        s = time.time()
        self.stitch()
        print("10. stitching time: ", time.time()-s)


    def get_inlay_outer(self) -> open3d.geometry.TriangleMesh:
        return self.inlay_outer
    
    def get_stitched_inlay(self) -> open3d.geometry.TriangleMesh:
        return self.stitched_inlay