import copy
import multiprocessing
import os
import random
import sys
import time

import numpy as np
import open3d as o3d
import trimesh
import trimesh.proximity
from scipy.spatial import KDTree

from slice_mesh import slice_mesh
from thickness import inlayAdjust
from utils import read_mesh

sys.path.append(".")
sys.path.append("..")
import tps
from tps import tps_runner
from utils import compute_signed_distance, get_distance


class MeshRegistration:
    def __init__(self, configs):
        self.draw = configs["draw"]
        self.isSave = configs["isSave"]
        self.path = configs["path"]
        self.sub_path = configs["sub_path"]
        self.teeth_num = configs["teeth_num"]
        self.registAlph = configs["registAlph"]
        self.tpsAlpha = configs["tpsAlpha"]
        self.normalSize = configs["normalSize"]
        self.cement_gap = configs["cement_gap"]
        self.height_of_minimal_gap = configs["height_of_minimal_gap"]
        self.height_of_minimal_gap2 = configs["height_of_minimal_gap2"]
        self.minimal_gap = configs["minimal_gap"]
        self.lastSubmehs_tps_dis = configs["lastSubmehs_tps_dis"]
        self.stitching_width = configs["stitching_width"]
        self.continueFlag = configs["continueFlag"]
        self.thickness = configs["thickness"]
        self.boundary_protect_range_offset = configs["boundary_protect_range_offset"]
        self.boundaryTPS_range = configs["boundaryTPS_range"]
        self.outer_pre_lift_dis = configs["outer_pre_lift_dis"]
        self.skiped = False
        self.save_path = configs["save_path"]
        self.lib_tooth_configs = configs["tooth_lib_configs"]
        self.adjust_crown = configs["adjust_crown"]

    def calcuPerPoint(self, p, a, b):
        v1 = p - a
        v2 = b - a
        n = np.dot(v2, v2)
        if n > 0:
            dot = np.dot(v1, v2)
            if dot <= 0:
                return a
            if dot >= n:
                return b
            p_ = a + (dot / n) * v2
            return p_
        return b

    def tri2o3d(self, mesh_trimesh):
        vertices = np.asarray(mesh_trimesh.vertices)
        faces = np.asarray(mesh_trimesh.faces)
        vertex_normals = copy.deepcopy(np.asarray(mesh_trimesh.vertex_normals))
        mesh_open3d = o3d.geometry.TriangleMesh()
        mesh_open3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_open3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_open3d.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
        return mesh_open3d

    def o3d2tri(self, mesh_open3d):
        if len(mesh_open3d.vertices) == 0 or len(mesh_open3d.triangles) == 0:
            raise ValueError("Mesh has no vertices or triangles")
        vertices = np.asarray(mesh_open3d.vertices)
        faces = np.asarray(mesh_open3d.triangles)
        mesh_trimesh = trimesh.Trimesh(vertices, faces)
        if len(mesh_open3d.vertex_normals) != 0:
            mesh_trimesh.vertex_normals = np.asarray(mesh_open3d.vertex_normals)
        return mesh_trimesh

    def reverse_trimesh(self, mesh):
        temp_faces = np.asarray(mesh.faces)
        reversed_faces = np.column_stack(
            [temp_faces[:, 1], temp_faces[:, 0], temp_faces[:, 2]]
        )
        reversed_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=reversed_faces)
        return reversed_mesh

    def doTPS(self, mesh, pcd, last_index, last_points, alph):
        """
        perform TPS deformation to adapt the mesh/pcd from last_points[:,0] to last_points[:,1]
        """
        last_point_dst = (
            alph * (last_points[:, 1] - last_points[:, 0]) + last_points[:, 0]
        )
        trans = tps.TPS(
            np.asarray(pcd.points)[last_index], last_point_dst, self.tpsAlpha
        )
        transformed_xyz = trans(np.asarray(pcd.points))
        tps_mesh = trimesh.Trimesh(transformed_xyz, mesh.faces)
        pcd.points = o3d.utility.Vector3dVector(tps_mesh.vertices)
        if tps_mesh.vertex_normals.flags.writeable:
            pcd.normals = o3d.utility.Vector3dVector(tps_mesh.vertex_normals)
        else:
            pcd.normals = o3d.utility.Vector3dVector(
                np.array(tps_mesh.vertex_normals, copy=True)
            )
        return tps_mesh

    def calcuPcdNormal(self, mesh):
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals
        return pcd.normals

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def draw_tps_result(self, points, pcdList):
        lines, point, i_ = [], [], 0
        c = 0
        for i in pcdList:
            i.paint_uniform_color([0.3, 0.3, 0.3 + c * 0.3])
            c += 1

        for p0, p1_ in points:
            lines.append([2 * i_, 2 * i_ + 1])
            point.append(p0)
            point.append(p1_)
            i_ += 1
        point = np.array(point)
        color = [[1, 0, 0] for i in range(len(lines))]
        # draw line
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(color)
        lines_pcd.points = o3d.utility.Vector3dVector(point)
        o3d.visualization.draw_geometries(pcdList + [lines_pcd])

    def getNoLine(self, target, source, normalSize=2.0):
        """
        估计目标和源点云的法线，计算目标上的关键点，并使用KDTree搜索找到源点云上的最近点。
        如果`draw`属性设置为True，则可选择绘制结果。

        参数:
        target (o3d.geometry.PointCloud): 目标点云。
        source (o3d.geometry.PointCloud): 源点云。
        normalSize (float, optional): 用于法线估计和KDTree搜索的半径。默认值为2.0。

        返回:
        tuple: 包含以下内容的元组:
            - lastIndex (list): 源点云中最近点的索引。
            - lastPoints (np.ndarray): 点及其投影的数组。
        """
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=normalSize, max_nn=int(30 * normalSize)
        )
        target.estimate_normals(search_param=search_param)
        source.estimate_normals(search_param=search_param)
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(target)
        # keypoints = target.farthest_point_down_sample(500)
        key_map = []
        pcd_tree1 = o3d.geometry.KDTreeFlann(target)
        for p in np.asarray(keypoints.points):
            [k, idx, _] = pcd_tree1.search_knn_vector_3d(p, 1)
            key_map.append(idx[0])

        # calcu nearest point using kdtree
        pcd_tree2 = o3d.geometry.KDTreeFlann(source)
        lastIndex, lastPoints, points = [], [], []
        for i in key_map:
            p0 = target.points[i]
            nv = target.normals[i]
            [k, idx, _] = pcd_tree2.search_radius_vector_3d(p0, 1.0)
            disMap = {}
            for j in idx:
                p = source.points[j]
                v = p - p0
                vp = np.dot(nv, v) * nv + p0
                pdis = np.dot(p - vp, p - vp)
                disMap[j] = pdis
            sorted_items = sorted(disMap.items(), key=lambda item: item[1])
            if len(sorted_items) > 0:
                index = sorted_items[0][0]
                ps = source.points[index]
                nvs = source.normals[index]
                vs = p0 - ps
                pl = ps + nvs * np.dot(nvs, vs)
                lastIndex.append(index)
                tempPoints = [ps, pl]
                lastPoints.append(tempPoints)
                if self.draw:
                    points.append(tempPoints)
        if self.draw:
            self.draw_tps_result(points, [target, source])
        return (lastIndex, np.asarray(lastPoints))

    def register_icpP2P(self, source, target):
        threshold = 0.2 * np.max(source.get_max_bound() - source.get_min_bound())
        trans_init = np.eye(4)
        # if self.draw: self.draw_registration_result(source, target, trans_init)
        # evaluation = o3d.pipelines.registration.evaluate_registration(
        #     source, target, threshold, trans_init
        # )
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
        )
        # if self.draw :self.draw_registration_result(source, target, reg_p2p.transformation)
        return reg_p2p

    def register_icpP2F(self, source, target):
        trans_init = np.eye(4)
        if self.draw:
            self.draw_registration_result(source, target, trans_init)
        threshold = 0.1 * np.max(source.get_max_bound() - source.get_min_bound())
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.5 * threshold, max_nn=100
            )
        )
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
        )
        if self.draw:
            self.draw_registration_result(source, target, reg_p2l.transformation)
        return reg_p2l
    
    def poisson_reconstruct(self, mesh, outer):
        """
        Reconstruct a surface to segment(slice) the lib tooth mesh.
        Args:
            mesh (_type_): _description_

        Returns:
            _type_: _description_
        """

        mesh.compute_vertex_normals()

        bound_pts, norms = self.getBoundaryPoints(mesh)
        bound_pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.array(bound_pts))
        )
        
        prox = trimesh.proximity.ProximityQuery(outer)
        signed_dis = prox.signed_distance(np.array(mesh.vertices))
        if max(signed_dis) > -0.5:
            normals = np.array(norms)
            normals = normals / np.linalg.norm(normals)
            
            normals_ = np.mean(np.array(bound_pts)) - np.array(bound_pts)
            normals_ = normals_ / np.linalg.norm(normals_)
            
            normals = normals  + normals_
            normals = normals / np.linalg.norm(normals)
            
            bound_pc.normals = o3d.utility.Vector3dVector(normals)
            
            idx = get_distance(self.o3d2tri(mesh), np.asarray(
                        mesh.get_non_manifold_edges(allow_boundary_edges=False)
                    ).flatten().tolist(), 2.5)
            mesh.remove_vertices_by_index(idx)
            
            mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
            mesh = mesh.split(only_watertight=False)
            mesh = mesh[np.argmax(np.array([x.vertices.shape[0] for x in mesh]))]
            
            mesh = mesh.as_open3d
            mesh.compute_vertex_normals()
            pcd_q = o3d.geometry.PointCloud(mesh.vertices)
            pcd_q.normals = mesh.vertex_normals
            
            pcd_q += bound_pc
        else:
            bound_pc.normals = o3d.utility.Vector3dVector(np.array(norms))
            pcd_q = o3d.geometry.PointCloud(mesh.vertices)
            pcd_q.normals = mesh.vertex_normals
            pcd_q = pcd_q.random_down_sample(
                sampling_ratio=0.05
            )  # down sample the point cloud

            pcd_q += bound_pc  # but keep all the boundary points
            pcd_q = pcd_q.remove_duplicated_points()

        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_q,
            depth=7,
            scale=1.5,
            linear_fit=False,
            n_threads=multiprocessing.cpu_count(),
        )[0]
        poisson_mesh.compute_vertex_normals()
        
        return poisson_mesh

    def calcuBsubQ(self, mesh_b, mesh_q):
        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = mesh_b.vertices
        pcd_q = o3d.geometry.PointCloud()
        pcd_q.points = mesh_q.vertices

        pcd_tree = o3d.geometry.KDTreeFlann(pcd_b)
        boundary_q_index = set()
        for p in pcd_q.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            boundary_q_index.add(idx[0])

        pcd_b_q = o3d.geometry.PointCloud()
        for i in range(len(pcd_b.points)):
            if i not in boundary_q_index:
                pcd_b_q.points.append(pcd_b.points[i])

        # calcu boundary point index
        boundary_q_index = list(
            set(
                np.array(
                    mesh_q.get_non_manifold_edges(allow_boundary_edges=False)
                ).flatten()
            )
        )
        boundarySet = []
        for i in boundary_q_index:
            boundarySet.append(pcd_q.points[i])
        return (pcd_b_q, boundarySet)

    def read_o3d_mesh(self, path):
        mesh = trimesh.load(path)
        vertices = mesh.vertices
        faces = mesh.faces
        mesh_open3d = o3d.geometry.TriangleMesh()
        mesh_open3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_open3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_open3d.compute_vertex_normals()
        return mesh_open3d

    def meshSub(self, mesh_a, mesh_b):
        a_vertices = np.asarray(mesh_a.vertices).tolist()
        b_vertices = np.asarray(mesh_b.vertices).tolist()
        for i in range(len(b_vertices)):
            b_vertices[i] = tuple(b_vertices[i])
        b_vertices = set(b_vertices)
        delIndex = []
        for i in range(len(a_vertices)):
            if tuple(a_vertices[i]) in b_vertices:
                delIndex.append(i)

        o3d_mesh_a = self.tri2o3d(mesh_a)
        o3d_mesh_a.remove_vertices_by_index(delIndex)
        return o3d_mesh_a

    def liftingMesh(self, mesh, liftDis, initDis, redius):
        boundary_index = list(
            set(
                np.array(
                    mesh.get_non_manifold_edges(allow_boundary_edges=False)
                ).flatten()
            )
        )

        pcd_b = o3d.geometry.PointCloud()
        for i in boundary_index:
            pcd_b.points.append(mesh.vertices[i])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_b)
        redius_Distance_list = []
        for i in range(len(mesh.vertices)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(mesh.vertices[i], 1)
            d = np.linalg.norm(np.asarray(pcd_b.points[idx[0]] - mesh.vertices[i]))
            redius_Distance_list.append(d)
        redius_Distance_array = np.asarray(redius_Distance_list)
        redius_Distance_array = np.where(
            redius_Distance_array < redius, redius_Distance_array / redius, 1
        )
        liftDis = (redius_Distance_array * (liftDis - initDis) + initDis).reshape(
            redius_Distance_array.shape[0], 1
        )

        vertices = np.asarray(mesh.vertices)
        vertices_normals = self.calcuPcdNormal(mesh)
        # mesh.compute_vertex_normals()
        normals = np.asarray(vertices_normals)
        vertices = vertices + liftDis * normals
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

    def cement(self, h):
        if h < self.height_of_minimal_gap2:
            out = self.minimal_gap
        elif h > self.height_of_minimal_gap:
            out = self.cement_gap
        else:
            out = self.cement_gap - (self.height_of_minimal_gap - h) / (
                self.height_of_minimal_gap - self.height_of_minimal_gap2
            ) * (self.cement_gap - self.minimal_gap)
        return out

    def get_cement_gap(self, mesh_q):
        """
        备牙膨胀，用于留出填充间隙。
        首先修复网格的法线，然后遍历网格的每个顶点。
        对于每个顶点，计算其法线和相邻顶点的法线，如果角度大于60度，不对该点做调整。
        反之，使用法线移动该顶点。
        处理所有顶点后，使用TPS对网格进行重采样。

        """
        # TODO apply the following parameters
        # self.cement_gap_spacing, self.cement_gap_spacing_boundary, self.q_lift_radius

        # mesh_q.fix_normals()
        print("Processing cement gap")
        if len(mesh_q.vertices) == 0 or len(mesh_q.faces) == 0:
            raise ValueError("Mesh has no vertices or faces")

        if len(mesh_q.vertex_normals) == 0:
            raise ValueError("Mesh has no vertex normals")
        if len(mesh_q.vertices) > 10000:
            mesh_q = mesh_q.simplify_quadratic_decimation(face_count=10000)

        try:
            outlines = mesh_q.outline().referenced_vertices
        except Exception as e:
            raise ValueError("Failed to compute mesh outline") from e
        if len(outlines) == 0:
            raise ValueError("Mesh outline has no vertices")
        # 构建 KD 树
        try:
            tree = KDTree(mesh_q.vertices[outlines])
        except Exception as e:
            raise ValueError("Failed to build KDTree for mesh outline") from e
        # 计算每个点到边缘点云的最小距离
        try:
            distances, _ = tree.query(mesh_q.vertices)
        except Exception as e:
            raise ValueError("Failed to query distances from KDTree") from e
        points = []
        n = []
        for k in range(len(mesh_q.vertices)):
            if k in outlines:
                points.append(mesh_q.vertices[k])
                continue
            a = mesh_q.vertex_normals[k]
            if len(mesh_q.vertex_neighbors[k]) == 0:
                raise ValueError(f"Vertex {k} has no neighbors")
            b = mesh_q.vertex_normals[mesh_q.vertex_neighbors[k]]
            try:
                cos_angles = [np.clip(np.dot(a, x), -1, 1) for x in b]
                angle = np.min(np.arccos(cos_angles) / np.pi * 180)
            except Exception as e:
                print(f"Error computing angles for vertex {k}: {e}")
                angle = 61
            if angle > 60:
                n.append(k)
                points.append(mesh_q.vertices[k])
                continue
            else:
                v_n = a
            points.append(mesh_q.vertices[k] + v_n * self.cement(distances[k]))
        try:
            points_id = random.sample(
                [x for x in range(len(mesh_q.vertices))],
                len(mesh_q.vertices) // 4,
            )
            points_id = [x for x in points_id if x not in n]
        except Exception as e:
            raise ValueError("Failed to sample points for TPS") from e
        if len(points_id) == 0:
            raise ValueError("No valid points for TPS")
        points_tps = np.array(points)[points_id]
        try:
            print("Running TPS transformation")
            print(len(mesh_q.vertices), "\t", len(points_id), "\t", len(points_tps))
            self.inner_dilation = tps_runner(mesh_q, points_id, points_tps)
        except Exception as e:
            raise ValueError("Failed to run TPS transformation") from e

    def doLastMeshInsertTps(self, mesh_source, boundarySet):
        """
        Params:
            mesh_source: Trimesh mesh to be sliced
            boundarySet: a set of boundary points in xyz coordinates
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(mesh_source.vertices))
        pcd.normals = o3d.utility.Vector3dVector(np.array(mesh_source.vertex_normals))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # sub sample the boundary points as very dense points into sampleIndexList
        lastIndexSet = set()
        sampleIndexList = []
        sampleDis = 0.01
        p0 = boundarySet[0]
        for i in range(len(boundarySet)):
            p1 = boundarySet[i]
            v = p1 - p0
            normal = np.linalg.norm(v)
            if normal < sampleDis:
                continue
            vn = v / normal
            num = int(normal / sampleDis)
            vn = vn * sampleDis
            for j in range(num):
                p = p0 + vn * j
                sampleIndexList.append(p)
            p0 = p1
        # if a vert is close to the sampled boundary poitns, we add it to lastIndexSet
        for p in sampleIndexList:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            px = pcd.points[idx[0]]
            [__, idx_, _] = pcd_tree.search_knn_vector_3d(px, 3)
            for j in idx_:
                if j not in lastIndexSet:
                    lastIndexSet.add(j)
        # if triangle contains a vert that is in lastIndexSet, then add it to lastFaces
        faces = mesh_source.faces
        lastFaces = []
        for face_idx in faces:
            flag = True
            for i in face_idx:
                if i in lastIndexSet:
                    flag = False
            if flag:
                lastFaces.append(face_idx)

        lastMesh = trimesh.Trimesh(
            vertices=mesh_source.vertices, faces=np.asarray(lastFaces)
        )
        if self.draw:
            lastMesh.show()
        mesh = self.tri2o3d(lastMesh)
        mesh.compute_vertex_normals()
        # remove the largest connected component, assuming the inlay is smaller than the other half
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        mesh_0 = copy.deepcopy(mesh)
        lagest_area_idx = cluster_area.argmax()
        triRemove = triangle_clusters == lagest_area_idx
        mesh_0.remove_triangles_by_mask(triRemove)
        # V = np.asarray(mesh_0.vertices)
        # F = np.asarray(mesh_0.triangles)
        # new_mesh = trimesh.Trimesh(vertices=V, faces=F)
        # o3d_new_mesh = self.tri2o3d(new_mesh)
        # o3d_new_mesh = self.boundarySmooth(o3d_new_mesh)
        # return o3d_new_mesh

        mesh_0.remove_unreferenced_vertices()
        return copy.deepcopy(mesh_0)

    def fstBoudaryTpsVec(self, mesh_source, boundarySet, alph):
        # get boundary non line vector
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_source.vertices))
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=1000)
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(30)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        lastIndexMap = {}
        search_num = int(100)
        for p in boundarySet:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, search_num)
            nv = pcd.normals[idx[0]]
            for j in idx:
                ps = pcd.points[j]
                vs = p - ps
                dot = np.dot(nv, vs)
                v = dot * nv
                d = np.linalg.norm(v)
                if d > 0.1:
                    if j not in lastIndexMap:
                        lastIndexMap[j] = v
                    else:
                        lastIndexMap[j] = 0.5 * (lastIndexMap[j] + v)
        if len(lastIndexMap) < int(0.25 * len(boundarySet)):
            return (True, mesh_source)
        lastIndex, lastPoint = [], []
        points = []
        for i in lastIndexMap:
            lastIndex.append(i)
            lastPoint.append([pcd.points[i], pcd.points[i] + lastIndexMap[i]])
            if self.draw:
                points.append([pcd.points[i], pcd.points[i] + lastIndexMap[i]])
        if self.draw:
            self.draw_tps_result(points, [pcd])
        lastPoint_array = np.asarray(lastPoint)
        result = self.doTPS(mesh_source, pcd, lastIndex, lastPoint_array, alph).copy()
        return (False, result)

    def lastBoudaryTpsVec(self, mesh_source, boundarySet, alph, mult):
        # get boundary non line vector
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_source.vertices))
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=1000)
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(30)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        lastIndexMap = {}
        search_num = int(20 * mult)
        for p in boundarySet:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, search_num)
            nv = pcd.normals[idx[0]]
            for j in idx:
                ps = pcd.points[j]
                vs = p - ps
                dot = np.dot(nv, vs)
                v = dot * nv
                d = np.linalg.norm(v)
                if d > 0.1:
                    if j not in lastIndexMap:
                        lastIndexMap[j] = v
                    else:
                        lastIndexMap[j] = 0.5 * (lastIndexMap[j] + v)
        if len(lastIndexMap) < int(0.25 * len(boundarySet)):
            return (True, mesh_source)
        lastIndex, lastPoint = [], []
        points = []
        for i in lastIndexMap:
            lastIndex.append(i)
            lastPoint.append([pcd.points[i], pcd.points[i] + lastIndexMap[i]])
            if self.draw:
                points.append([pcd.points[i], pcd.points[i] + lastIndexMap[i]])
        if self.draw:
            self.draw_tps_result(points, [pcd])
        lastPoint_array = np.asarray(lastPoint)
        result = self.doTPS(mesh_source, pcd, lastIndex, lastPoint_array, alph).copy()
        return (False, result)  # (lastIndex, np.array(lastPoint))

    def doBoundaryTps(self, mesh_source, boundarySet, alph=0.3, iterNum=10):
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(mesh_source.vertices)
        if mesh_source.vertex_normals.flags.writeable:
            pcd_source.normals = o3d.utility.Vector3dVector(mesh_source.vertex_normals)
        else:
            pcd_source.normals = o3d.utility.Vector3dVector(
                np.array(mesh_source.vertex_normals, copy=True)
            )
        mult = 1.0
        mult_lev_min = 0.2
        for i in range(iterNum):
            mult = mult_lev_min * mult + (1 - mult_lev_min) * mult * (
                iterNum - 1 - i
            ) / (iterNum - 1)
            alph = alph + (1 - alph) * (i) / (iterNum - 1)
            flag, mesh_tps = self.lastBoudaryTpsVec(
                mesh_source, boundarySet, alph, mult
            )
            mesh_source = mesh_tps.copy()
            if flag:
                break
            if i > 3:
                break
        return mesh_source

    def getBoundaryPoints(self, mesh):
        """get the boundary points of a mesh IN THE ORDER of how they appear in the mesh"""
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
            mesh
        )
        boundary_point = []
        normals = []
        for boundary in half_edge_mesh.get_boundaries():
            for vertex_id in boundary:
                boundary_point.append(half_edge_mesh.vertices[vertex_id])
                normals.append(half_edge_mesh.vertex_normals[vertex_id])
        return boundary_point, normals

    def sampleMesh(self, mesh, num):
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=num)
        pcd_mesh = o3d.geometry.PointCloud()
        pcd_mesh.points = o3d.utility.Vector3dVector(mesh.vertices)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_mesh)
        boundary_index = set()
        for p in pcd.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            boundary_index.add(idx[0])
        return boundary_index

    def getPcdKeyPoints(
        self,
        pcd,
        salient_radius=0.0,
        non_max_radius=0.0,
        gamma_21=0.975,
        gamma_32=0.975,
        min_neighbors=5,
    ):
        """
        This method calls o3d.geometry.keypoint.compute_iss_keypoints to find the iss keypoint indices of a pc
        """
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
            pcd,
            salient_radius=salient_radius,
            non_max_radius=non_max_radius,
            gamma_21=gamma_21,
            gamma_32=gamma_32,
            min_neighbors=min_neighbors,
        )
        """
        o3d.geometry.keypoint.compute_iss_keypoints()

        "Function that computes the ISS keypoints from an input point cloud. 
        This implements the keypoint detection modules proposed in Yu Zhong, 
        'Intrinsic Shape Signatures: A Shape Descriptor for 3D Object Recognition', 2009."
        
        Parameters:
            input: "The Input point cloud."
            salient_radius = 0.0: "The radius of the spherical neighborhood used to detect keypoints"
            non_max_radius = 0.0: "The non maxima suppression radius"
            gamma_21 = 0.975: "The upper bound on the ratio between the second and the first eigenvalue returned by the EVD"
            gamma_32 = 0.975: "The upper bound on the ratio between the third and the second eigenvalue returned by the EVD"
            min_neighbors = 5: "Minimum number of neighbors that has to be found to consider a keypoint"

        Returns:
            keypoints: "The output keypoints as a point cloud."
        """
        liftIndex = []
        for p in keypoints.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            liftIndex.append(idx[0])
        return liftIndex

    def calcuMesh2Pcd(self, mesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        mesh.compute_vertex_normals()
        pcd.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
        return pcd

    def subMeshBoundaryTps(self, dstboundary, sub_mesh, iterNum=2):
        sub_mesh.remove_degenerate_triangles()
        sub_mesh.remove_duplicated_vertices()
        sub_mesh.remove_unreferenced_vertices()

        mesh = self.o3d2tri(sub_mesh)
        pcd_bound = o3d.geometry.PointCloud()
        for p in dstboundary:
            pcd_bound.points.append(p)
        bound_tree = o3d.geometry.KDTreeFlann(pcd_bound)  # inner margin pad tree

        sub_pc = o3d.geometry.PointCloud(sub_mesh.vertices)
        sub_pc.normals = sub_mesh.vertex_normals
        sub_pc_tree = o3d.geometry.KDTreeFlann(sub_pc)  # outer points tree
        outer_bound_pid = list(
            set(
                np.array(
                    sub_mesh.get_non_manifold_edges(allow_boundary_edges=False)
                ).flatten()
            )
        )
        outer_bound_pc = sub_pc.select_by_index(outer_bound_pid)

        while iterNum > 0:
            iterNum -= 1
            pcd = self.calcuMesh2Pcd(self.tri2o3d(mesh))

            boundary_index = self.getBoundSet(self.tri2o3d(mesh))
            index, points = [], []
            for i in boundary_index:  # add boundary keypoints to move the margin
                p = pcd.points[i]  # the current outer margin point
                [k, idx, _] = bound_tree.search_knn_vector_3d(
                    p, 1
                )  # find the closest point on the margin of inner (Q)
                L = len(dstboundary)
                d = -1
                p0 = p
                for i_ in range(idx[0] - 3, idx[0] + 3):
                    j0 = i_
                    j1 = i_ + 1
                    if j0 >= L:
                        j0 = i_ - L
                    if j1 >= L:
                        j1 = j1 - L
                    p1 = pcd_bound.points[j0]
                    p2 = pcd_bound.points[j1]

                    p_ = self.calcuPerPoint(p, p1, p2)
                    d_ = np.linalg.norm(p_ - p)
                    if d < 0:
                        d = d_
                        p0 = p_
                    elif d_ < d:
                        d = d_
                        p0 = p_  # p0 is the inner margin point for the current outer margin point, p
                # move p towards p0
                vl = p0 - p
                dl = np.linalg.norm(vl)
                pl = p + vl * ((dl - 0.7 * self.stitching_width) / dl)
                # move p0 (the point on the margin of inner) towards its normal direction for a little bit (0.3 * self.stitching_width)
                [k, idx, _] = sub_pc_tree.search_knn_vector_3d(p0, 5)
                p0_norm = np.sum(np.asarray(sub_pc.normals)[idx], axis=0)
                p0_norm /= np.linalg.norm(p0_norm)
                pl += 0.3 * self.stitching_width * p0_norm
                # save the ids and dst positions
                index.append(i)
                points.append([p, pl])
            tempSet = set(
                boundary_index
            )  # add boundary verts to the index list for stretching the margin

            # To keep the non-boundary points stay in the same place is to use all verts far from the outer margin
            if len(sub_pc.points) > 1000:
                choice_list = np.random.choice(
                    np.asarray(sub_pc.points).shape[0], size=1000, replace=False
                )
                sp_points = np.asarray(sub_pc.points)[choice_list]
            else:
                choice_list = []
                sp_points = np.asarray(sub_pc.points)
            for i, p in enumerate(
                sp_points
            ):  # keep the non-boundary points stay in the same place
                if len(choice_list):
                    i = choice_list[i]
                if i in tempSet:
                    continue
                pcd_source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([p]))
                if (
                    pcd_source.compute_point_cloud_distance(outer_bound_pc)[0]
                    < self.boundaryTPS_range
                ):
                    continue
                index.append(i)
                points.append([p, p])

            if self.draw:
                self.draw_tps_result(points, [pcd, pcd_bound])
            points = np.asarray(points)
            if self.isSave:
                pts1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, 0]))
                pts1.paint_uniform_color([1, 0, 0])
                pts2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, 1]))
                pts2.paint_uniform_color([0, 0, 1])
                o3d.io.write_point_cloud(
                    os.path.join(
                        self.save_path, f"6.1_bound_tps_src_{self.teeth_num}.ply"
                    ),
                    pts1,
                )
                o3d.io.write_point_cloud(
                    os.path.join(
                        self.save_path, f"6.2_bound_tps_dst_{self.teeth_num}.ply"
                    ),
                    pts2,
                )
            mesh = self.doTPS(mesh, pcd, index, points, 1.0)

        return mesh

    def getPcdNormalsByMesh(self, pcd, mesh, num=100, radiu=0.5):
        mesh.compute_vertex_normals()
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radiu, max_nn=num)
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(num)
        sampIndex = self.sampleMesh(mesh, 10)
        total = 0
        for i in sampIndex:
            total += np.dot(pcd.normals[i], mesh.vertex_normals[i])
        if total < 0:
            pcd.normals = o3d.utility.Vector3dVector(-1.0 * np.asarray(pcd.normals))
        return pcd.normals

    def getBoundSet(self, mesh):
        half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
            mesh
        )
        boundary_index = []
        for boundary in half_edge_mesh.get_boundaries():
            for i in boundary:
                boundary_index.append(i)
        return boundary_index

    def doSubMeshTps(self, dstboundary, sub_mesh, liftDis=0.5):
        tps_mesh = self.o3d2tri(sub_mesh)

        next_mesh = sub_mesh
        next_mesh.compute_vertex_normals()
        nextPcd = o3d.geometry.PointCloud()
        nextPcd.points = o3d.utility.Vector3dVector(next_mesh.vertices)
        nextPcd.normals = next_mesh.vertex_normals

        tree = o3d.geometry.KDTreeFlann(nextPcd)

        liftIndex = []

        # keypoints = o3d.geometry.keypoint.compute_iss_keypoints(nextPcd)
        init_dis = 1.0
        init_keypoints_idx = np.array([], dtype=np.int64)
        boundary_pts = np.asarray(sub_mesh.vertices)[self.getBoundSet(sub_mesh)]
        boundary_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boundary_pts))
        dis_to_boundary = np.asarray(nextPcd.compute_point_cloud_distance(boundary_pc))
        while True:
            init_keypoints_idx = np.where(dis_to_boundary > init_dis)[0]
            keypoints = nextPcd.select_by_index(init_keypoints_idx)
            key_tree = KDTree(np.asarray(keypoints.points))
            distances, _ = key_tree.query(boundary_pts)
            dist_idx = np.where(distances > 2)[0]
            if len(dist_idx):
                if init_dis < 0.5:
                    break
                init_dis -= 0.1
            else:
                break
            
        if len(np.asarray(keypoints.points)) > 100:
            keypoints = keypoints.farthest_point_down_sample(100)

        for p in keypoints.points:
            [k, idx, _] = tree.search_knn_vector_3d(p, 1)
            liftIndex.append(idx[0])

        nextPcd.normals = self.getPcdNormalsByMesh(nextPcd, next_mesh, 200, 1.0)
        index, points = [], []
        tempSet = set(index)
        for i in liftIndex:
            if i in tempSet:
                continue
            p = nextPcd.points[i]
            n = nextPcd.normals[i]
            p1 = p + liftDis * n
            index.append(i)
            points.append([p, p1])

        points = np.asarray(points)
        tps_mesh_ = self.doTPS(
            tps_mesh, nextPcd, index, points, self.outer_pre_lift_dis
        )  # lift the outer for 1 mm

        mesh = self.subMeshBoundaryTps(
            dstboundary, self.tri2o3d(tps_mesh_)
        )  # strech the outer to meet Q margin
        mesh = self.boundarySmooth(self.tri2o3d(mesh))
        return self.o3d2tri(mesh)

    def doLastSubTps(self, dstboundary, sub_mesh):
        pcd_bound = o3d.geometry.PointCloud()
        for p in dstboundary:
            pcd_bound.points.append(p)

        pcd = self.calcuMesh2Pcd(sub_mesh)

        bound_tree = o3d.geometry.KDTreeFlann(pcd_bound)
        sub_o3d_mesh = self.tri2o3d(sub_mesh)

        boundary_index = self.getBoundSet(sub_o3d_mesh)

        index, points = [], []
        for i in boundary_index:
            p = pcd.points[i]
            [k, idx, _] = bound_tree.search_knn_vector_3d(p, 1)
            p0 = pcd_bound.points[idx[0]]
            p1 = pcd_bound.points[idx[1]]
            v0 = p0 - p
            v1 = p1 - p
            v = 0.5 * (v0 + v1)
            d = np.linalg.norm(v)
            if d > 0.01:
                pdst = p + v
                index.append(i)
                points.append([p, pdst])

        if self.draw:
            self.draw_tps_result(points, [pcd])
        if len(points) > 0:
            points = np.asarray(points)
            tps_mesh = self.doTPS(sub_mesh, pcd, index, points, 1.0)

        pcd.normals = self.getPcdNormalsByMesh(pcd, sub_mesh, 30, 0.5)

        for i in range(len(tps_mesh.vertices)):
            tps_mesh.vertices[i] = tps_mesh.vertices[i] + 0.02 * pcd.normals[i]

        if self.draw:
            tps_mesh.show()
        return tps_mesh

    def doRegist(self, pcd_source, pcd_b_q, mesh, alph, iterNum=2):
        def extract_total_rotation_angle(transform_matrix):
            # 提取3x3的旋转矩阵
            R = transform_matrix[:3, :3]

            # 计算旋转矩阵的迹
            trace_R = np.trace(R)

            # 检查迹的值
            if np.isclose(trace_R, 3):
                return 0.0  # 旋转角度为0度
            elif np.isclose(trace_R, -1):
                return 180.0  # 旋转角度为180度
            else:
                # 计算总旋转角度
                theta = np.arccos((trace_R - 1) / 2)
                # 将弧度转换为度数
                theta_degrees = np.degrees(theta)
                return theta_degrees

        for i in range(2):
            # register
            reg_p2l = self.register_icpP2F(pcd_source, pcd_b_q)
            if extract_total_rotation_angle(reg_p2l.transformation) > 15:
                break
                # # 获取点云的边界范围
                # bbox1 = pcd_source.get_axis_aligned_bounding_box()
                # bbox2 = pcd_b_q.get_axis_aligned_bounding_box()

                # # 计算 x 和 z 方向上的缩放比例
                # scale_x = (bbox2.get_extent()[0]) / (bbox1.get_extent()[0])
                # scale_z = (bbox2.get_extent()[2]) / (bbox1.get_extent()[2])

                # # 创建一个 4x4 的缩放矩阵，只在 x 和 z 方向缩放
                # scale_matrix = np.eye(4)
                # scale_matrix[0, 0] = scale_x
                # scale_matrix[2, 2] = scale_z

                # # 应用变换矩阵
                # pcd_source.transform(scale_matrix)
            else:
                pcd_source.transform(reg_p2l.transformation)
            # get non line vector
            index, points = self.getNoLine(pcd_b_q, pcd_source, self.normalSize)
            mesh = mesh.apply_transform(reg_p2l.transformation)
            # tps
            mesh = self.doTPS(mesh, pcd_source, index, points, alph).copy()
            # mesh = tps_mesh.copy()
            pcd_source.points = o3d.utility.Vector3dVector(mesh.vertices)
        return mesh

    def boundarySmooth(self, o3d_mesh):
        boundIndex = self.getBoundSet(o3d_mesh)
        for i in range(len(boundIndex) - 3):
            i0, i1, i2 = boundIndex[i : i + 3]
            p0 = o3d_mesh.vertices[i0]
            p2 = o3d_mesh.vertices[i2]
            o3d_mesh.vertices[i1] = 0.5 * (p0 + p2)
        return o3d_mesh

    def boundaryDel(self, boundaryPoints, mesh):
        pcd_bound = o3d.geometry.PointCloud()
        for p in boundaryPoints:
            pcd_bound.points.append(p)
        bound_tree = o3d.geometry.KDTreeFlann(pcd_bound)

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        boundSet = self.getBoundSet(mesh)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        lastIndexSet = set()
        for i in boundSet:
            p = pcd.points[i]
            [k, idx, _] = bound_tree.search_knn_vector_3d(p, 1)
            v = p - pcd_bound.points[idx[0]]
            d = 2.0 * np.linalg.norm(v)
            [k_, idx_, _] = pcd_tree.search_radius_vector_3d(p, d)
            for j in idx_:
                lastIndexSet.add(j)

        faces = np.asarray(mesh.triangles)
        lastFaces = []

        for face_idx in faces:
            flag = True
            for i in face_idx:
                if i in lastIndexSet:
                    flag = False
            if flag:
                lastFaces.append(face_idx)
        lastMesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(lastFaces)
        )
        o3d_new_mesh = self.tri2o3d(lastMesh)
        o3d_new_mesh = self.boundarySmooth(o3d_new_mesh)
        return o3d_new_mesh

    def remove_small_connected_components(self, mesh):
        """
        Only keep the largest connected component
        Args:
            mesh: open3d.geometry.TriangleMesh

        Returns:
            noise_removed: open3d.geometry.TriangleMesh
        """
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        noise_removed = copy.deepcopy(mesh)
        triangles_to_remove = np.where(triangle_clusters != largest_cluster_idx)[
            0
        ].astype(np.int32)
        noise_removed.remove_triangles_by_index(triangles_to_remove)
        noise_removed.remove_unreferenced_vertices()
        return noise_removed

    def ensure_thickness(
        self,
        inlay_outer,
        inlay_inner,
        thickness=0.8,
        boundary_protect_range_offset=-0.2,
        lib_tooth_config=None,
    ):
        verts_outer = np.array(inlay_outer.vertices).astype(np.float64)
        faces_outer = np.array(inlay_outer.faces).astype(np.int32)
        verts_inner = np.array(inlay_inner.vertices).astype(np.float64)
        faces_inner = np.array(inlay_inner.faces).astype(np.int32)
        groove_vert_ids = lib_tooth_config["oc_points"]

        verts, faces = inlayAdjust(
            verts_outer,
            faces_outer,
            verts_inner,
            faces_inner,
            groove_vert_ids,
            thickness=thickness,
            boundary_protect_range_offset=boundary_protect_range_offset,
        )
        # print(verts, faces)

        inflated_inlay_outer = trimesh.Trimesh(verts, faces)
        return inflated_inlay_outer

    def run(self, mesh_source, mesh_target, mesh_q, anta_scan):
        mesh = self.o3d2tri(mesh_source)
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = mesh_source.vertices

        # register & tps==============================================================
        t1 = time.time()
        if self.adjust_crown:
            # mesh_q = mesh_q.simplify_quadric_decimation(5000)
            mesh_q.compute_vertex_normals()
            mesh_q.remove_duplicated_vertices()
            mesh_q.remove_non_manifold_edges()
            mesh_q.remove_duplicated_triangles()
            mesh_q.remove_degenerate_triangles()
            mesh_q.remove_unreferenced_vertices()
            pcd_b_q, _ = self.calcuBsubQ(mesh_target, mesh_q)  # B-Q
            mesh = self.doRegist(pcd_source, pcd_b_q, mesh, self.registAlph).copy()
        # from test_ransac import ransac_point_cloud_registration
        # tr, _ = ransac_point_cloud_registration(mesh.vertices, np.asarray(pcd_b_q.points))
        # mesh.apply_transform(tr)
        print("1. register time: ", time.time() - t1)
        # mesh = trimesh.load('0_lib_tooth_46_transformed.ply')

        if self.isSave:
            o3d.io.write_triangle_mesh(
                os.path.join(self.save_path, f"0_lib_tooth_{self.teeth_num}.ply"),
                mesh_source,
            )
            o3d.io.write_point_cloud(
                os.path.join(self.save_path, f"1_b_sub_q_{self.teeth_num}.ply"), pcd_b_q
            )
            mesh.export(
                os.path.join(self.save_path, f"2_registreation_{self.teeth_num}.stl")
            )
        # lifting mesh_q===============================================================
        t1 = time.time()
        # self.liftingMesh(mesh_q, self.cement_gap_spacing, self.cement_gap_spacing_boundary, self.q_lift_radius)  #mesh, liftDis, initDis, redius
        # self.inner_dilation = self.o3d2tri(mesh_q)
        self.get_cement_gap(self.o3d2tri(mesh_q))
        print("2. dilation time: ", time.time() - t1)
        if self.isSave:
            self.inner_dilation.export(
                os.path.join(
                    self.save_path, f"3_dilation_0.04-0.08_{self.teeth_num}.stl"
                )
            )

        # Boundary TPS=================================================================
        t1 = time.time()
        boundarySet, _ = self.getBoundaryPoints(mesh_q)
        has_boundary_mesh = self.doBoundaryTps(mesh, boundarySet)
        print("3. doBoundaryTps time: ", time.time() - t1)
        if self.isSave:
            has_boundary_mesh.export(
                (
                    os.path.join(
                        self.save_path, f"4.1_boundary_TPSed_{self.teeth_num}.stl"
                    )
                )
            )
        # first occlusion =================================================================
        t1 = time.time()
        if self.adjust_crown:
            q_pro = trimesh.proximity.ProximityQuery(has_boundary_mesh)
            dis, _ = q_pro.vertex(np.asarray(anta_scan.vertices))
            anta_scan_points = np.asarray(anta_scan.vertices)[np.where(dis < 10)[0]]
            sign_dis, _ = compute_signed_distance(has_boundary_mesh, anta_scan_points)
            if max(sign_dis) > 0 and max(sign_dis) < 0.5:
                has_boundary_mesh.apply_translation([0, -max(sign_dis), 0])
                print("3.1. first occlusion time: ", time.time() - t1)
                if self.isSave:
                    has_boundary_mesh.export(
                        (
                            os.path.join(
                                self.save_path, f"4.2_first_occlusion_{self.teeth_num}.stl"
                            )
                        )
                    )
                # Boundary TPS=================================================================
                t1 = time.time()
                has_boundary_mesh = self.doBoundaryTps(has_boundary_mesh, boundarySet)
                print("3.2 doBoundaryTps2 time: ", time.time() - t1)
                if self.isSave:
                    has_boundary_mesh.export(
                        (
                            os.path.join(
                                self.save_path, f"4.3_boundary_TPSed2_{self.teeth_num}.stl"
                            )
                        )
                    )
            # thickness=================================================================
            t1 = time.time()
            if self.adjust_crown:
                has_boundary_mesh = self.ensure_thickness(
                    has_boundary_mesh,
                    self.inner_dilation,
                    thickness=self.thickness,
                    boundary_protect_range_offset=self.boundary_protect_range_offset,
                    lib_tooth_config=self.lib_tooth_configs,
                )
                if self.isSave:
                    has_boundary_mesh.export(
                        os.path.join(self.save_path, f"5_inflated_outer_{self.teeth_num}.stl")
                    )
            print("4. thickness time: ", time.time() - t1)

        # split mesh with boundary=====================================================
        t1 = time.time()
        # build poisson_mesh
        # poisson_mesh = self.poisson_reconstruct(mesh_q)
        poisson_mesh = self.poisson_reconstruct(self.inner_dilation.as_open3d, has_boundary_mesh)
        print("5. poisson reconstruction time: ", time.time() - t1)
        if self.isSave:
            o3d.io.write_triangle_mesh(
                os.path.join(self.save_path, f"7_poisson_mesh_{self.teeth_num}.stl"),
                poisson_mesh,
            )
        t1 = time.time()
        # poisson_mesh = read_mesh(os.path.join(self.save_path, "poisson_mesh.stl"))
        sub_mesh = slice_mesh(has_boundary_mesh.as_open3d, poisson_mesh)
        print("6. slice_mesh time: ", time.time() - t1)
        if self.isSave:
            sub_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(
                os.path.join(self.save_path, f"8_submesh_{self.teeth_num}.stl"),
                sub_mesh,
            )
        
        # sub mesh export to boundary======================================================
        t1 = time.time()
        sub_mesh = self.doSubMeshTps(boundarySet, sub_mesh, self.lastSubmehs_tps_dis)
        print("7. sub mesh tps time: ", time.time() - t1)
        if self.isSave:
            sub_mesh.export(
                os.path.join(self.save_path, f"9_tps_submesh_{self.teeth_num}.stl")
            )
        # unit surface=====================================================================
        self.inner_dilation_inversed = self.reverse_trimesh(self.inner_dilation).copy()
        return (sub_mesh, self.inner_dilation_inversed)


def main():
    nameDic = {
        "07-03-1968": [47],
        "case 2": [25, 26, 27],
        "cb1b-6637": [36],
        "D2000_20230207-10921601__Raw Preparation scan": [37],
        "D2000_20230210-10924010__Raw Preparation scan": [26],
        "designer_Zheng_20230202_1158_增奇口腔_袁_10922266__Raw Preparation scan": [25],
        "designer_zheng_20230210_1641_陈贲10921124": [36],
        "__16809_160808142629__18615657098__2018-04-10_11-04_赵明珠__25": [25],
        "__16809_161216093401__13668809646__2018-04-10_11-08_左书玉__": [47],
        "__16809_170222150709__15966620585__2018-04-10_13-30_尹新芹__": [27],
        "__16809_170308143212__15666012120__2018-04-10_11-17_于磊__": [46],
        "__92837_170620151346__18553105077__2018-04-10_11-20_王更如__": [37],
        "__92837_170623093805__18853161295__2018-04-10_13-28_尹新芹__": [46],
        # "__16809_170222150709__15966620585__2018-04-10_13-30_尹新芹__": [27],
        "__92837_170623162024__18853108596__2018-04-10_17-13_赵明珠__": [16],
        "__92837_180119084740__15966634308__2018-04-10_13-07_赵明珠__": [17],
        "__92837_171219083436__15820092927__2018-04-10_13-59_林志勇__": [37],
        "__16809_160823175454__13793156233__2018-04-10_13-46_王宁__": [35],
        "__16809_170406102459__13465937836__2018-04-10_10-59_孙剑__": [46],
        "__92837_171031084140__15954780317__2018-04-10_11-10_王华薇__": [26],
        # "__92837_171219083436__15820092927__2018-04-10_13-59_林志勇__": [37],
        "__92837_180124083053__15169095577__2018-04-10_13-23_林志勇__": [27],
        "52e5-0124": [27],
        "97728_20220927_1535_王兆杰10902624": [37, 47],
        "97728_20221027_0835_张益琳10908618": [16],
        "trios2_20220930_1519_10903932": [45],  # 嵌体形态奇怪，
        "__16809_170206115614__15064055656__2018-04-10_13-20_左书玉__": [36],
        "6008-6825": [45],
    }

    path = "./inlays/"
    config = {
        "path": path,
        "draw": False,
        "isSave": True,
        "continueFlag": True,  # true: checking ./result/, if result already generated will skipped.
        "registAlph": 0.8,
        "tpsAlpha": 0.01,
        "normalSize": 0.5,
        "q_lift_dis": 0.02,  # Q mesh lift last dis
        "q_lift_initDis": 0.0,  # Q mesh boundary lift dis
        "q_lift_radius": 1.0,  # Q mesh lift radius
        "lastSubmehs_tps_dis": 0.1,
        "lastBoundary_width": 0.02,  # boundary width
    }

    for _, dirs, __ in os.walk(path):
        for dir in dirs:
            print(dir)
            if dir != "cb1b-6637":
                continue
            if dir not in nameDic:
                continue
            sub_path = dir
            for n in nameDic[dir]:
                teeth_num = str(n)
                config["sub_path"] = sub_path + "/"
                config["teeth_num"] = teeth_num
                config["save_path"] = path + sub_path + teeth_num + "_"
                save_path = path + "/result/" + sub_path + "__" + teeth_num
                target_path = path + config["sub_path"] + teeth_num + "B.stl"
                source_path = path + config["sub_path"] + teeth_num + "S_.stl"
                q_path = path + config["sub_path"] + teeth_num + "Q.stl"

                mesh_source = read_mesh(source_path)
                mesh_target = read_mesh(target_path)
                mesh_q = read_mesh(q_path)

                MR = MeshRegistration(config)
                U, D = MR.run(mesh_source, mesh_target, mesh_q)
                U.export(save_path + "_U_mesh.ply")
                unitMesh = U + D
                unitMesh.export(save_path + "_unit_mesh.ply")
        break


if __name__ == "__main__":
    main()
