import base64
import copy
import json
import os
import random

import DracoPy
import networkx as nx
import numpy as np
import open3d as o3d
import trimesh.boolean
import trimesh.remesh
import trimesh.repair
import pylfda
import trimesh
from pytransform3d.rotations import matrix_from_two_vectors
from scipy.spatial import KDTree, transform

import get_edges.prep_edge_smoothing
import pypruners
from dental_arch_curve import DAC
from find_thickness_points import find_changed_faces, find_subdivided_faces
from tps import tps
from tracked_trimesh import TrackedTrimesh
from undercut_util import get_insert_direction
from directional_undercut_filling import filling_undercut

# import pymeshrepairer


def read_mesh_bytes(buffer):
    if buffer is None:
        return None
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F)


def read_drc(file_name):
    """read .drc or .mq file
    Args:
        file_name ([a directory]): [description]
    Returns:
        verctics and faces in np.array (n,3), (m,3)
    """
    with open(file_name, "rb") as draco_file:
        file_content = draco_file.read()
    b64_bytes = base64.b64encode(file_content)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def write_mesh_bytes(mesh, preserve_order=False):
    # 设置 Draco 编码选项
    # encoding_test = DracoPy.encode_mesh_to_buffer(mesh.vertices, mesh.faces, preserve_order=True, quantization_bits=8,
    #                                               compression_level=10, colors=None)
    encoding_test = DracoPy.encode_mesh_to_buffer(
        mesh.vertices, mesh.faces, preserve_order=preserve_order
    )
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def remove_intersection(array1, array2):
    # 将每一行转换为元组
    array1_tuples = [tuple(row) for row in array1]
    array2_tuples = set(tuple(row) for row in array2)

    # 找到需要去除的行索引
    indices_to_remove = np.where([row not in array2_tuples for row in array1_tuples])[0]
    return indices_to_remove


def remove_boundary_faces(mesh, face_indices):
    """移除边界上的面"""
    edges = mesh.edges_sorted
    unique_edges = trimesh.grouping.group_rows(edges, require_count=1)
    boundary_edges = edges[unique_edges]
    boundary_vertices = np.unique(boundary_edges)
    boundary_vertices = get_distance(mesh, boundary_vertices.tolist(), 2)
    # boundary_vertices = get_neighbors(boundary_vertices, 2, mesh)
    # boundary_vertices = np.unique(np.array([y for x in np.array(boundary_vertices) for y in x]))
    boundary_face_mask = np.any(
        np.isin(mesh.faces[face_indices], boundary_vertices), axis=1
    )
    non_boundary_faces = face_indices[~boundary_face_mask]

    return non_boundary_faces


def rotate_vector_to_90_degrees(vec):
    """
    使 vec2 在 vec1 和 vec2 构成的平面内旋转，使得 vec1 和 vec2 最终夹角为90度。
    :param vec1: numpy array, 固定方向的向量
    :param vec2: numpy array, 需要旋转的向量
    :return: numpy array, 旋转后的 vec2
    """
    # 单位化向量
    vec1 = vec[0]
    vec2 = vec[1]
    vec1_normalized = vec1 / np.linalg.norm(vec1)
    vec2_normalized = vec2 / np.linalg.norm(vec2)

    # 计算两个向量的法向量
    normal_vector = np.cross(vec1_normalized, vec2_normalized)
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    # 计算当前夹角
    cos_theta = np.dot(vec1_normalized, vec2_normalized)
    current_angle = np.arccos(cos_theta)

    # 需要旋转的角度
    required_rotation_angle = np.pi / 2 - current_angle

    # 构建旋转矩阵
    K = np.array(
        [
            [0, -normal_vector_normalized[2], normal_vector_normalized[1]],
            [normal_vector_normalized[2], 0, -normal_vector_normalized[0]],
            [-normal_vector_normalized[1], normal_vector_normalized[0], 0],
        ]
    )

    rotation_matrix = (
        np.eye(3)
        + np.sin(required_rotation_angle) * K
        + (1 - np.cos(required_rotation_angle)) * np.dot(K, K)
    )

    # 旋转向量
    rotated_vec2 = np.dot(rotation_matrix, vec2_normalized) * np.linalg.norm(vec2)

    return np.array([vec1, rotated_vec2])


def rotate_from_axis(axis):
    """
    根据给出的y方向向量和x方向向量，转换坐标系，求出旋转矩阵
    """
    v_x = axis[1]
    v_y = axis[0]
    v_x = v_x / np.linalg.norm(v_x)
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.cross(v_x, v_y)
    v_z = v_z / np.linalg.norm(v_z)
    M = np.column_stack((v_x, v_y, v_z))

    M_inv = np.linalg.inv(M)
    T = np.eye(4)  # 创建一个4x4的单位矩阵
    T[:3, :3] = M_inv  # 将3x3的M_inv赋值到4x4矩阵的左上角
    return T


def remove_degenerate_faces(mesh):
    """
    Remove degenerate faces from a trimesh mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh with possible degenerate faces.

    Returns
    -------
    cleaned_mesh : trimesh.Trimesh
        Mesh with degenerate faces removed.
    """
    faces = mesh.faces
    vertices = mesh.vertices

    # Detect degenerate faces
    is_degenerate = np.any(faces[:, [0]] == faces[:, [1, 2]], axis=1) | np.any(
        faces[:, [1]] == faces[:, [2]], axis=1
    )

    # Select non-degenerate faces
    valid_faces = faces[~is_degenerate]

    # Create a new mesh with non-degenerate faces
    cleaned_mesh = trimesh.Trimesh(vertices=vertices, faces=valid_faces)

    return cleaned_mesh


def get_biggest_mesh(mesh):
    """
    获取备牙里最大的连通区
    """
    mesh_split = mesh.split(only_watertight=False)
    mesh = mesh_split[np.argmax(np.array([x.vertices.shape[0] for x in mesh_split]))]
    return mesh


def merge_items(items) -> trimesh.Trimesh:
    """合并连通体

    Args:
        items (list[trimesh.Trimesh]): 连通体列表

    Returns:
        trimesh.Trimesh: 合并成一个网格
    """
    vertices_list = [mesh.vertices for mesh in items]
    faces_list = [mesh.faces for mesh in items]
    faces_offset = np.cumsum([v.shape[0] for v in vertices_list])
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    vertices = np.vstack(vertices_list)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

    merged_meshes = trimesh.Trimesh(vertices, faces)
    return merged_meshes


def obb_mesh(mesh: trimesh.Trimesh, x_axis: np.array) -> trimesh.Trimesh:
    """找出三颗牙模型的OBB框，保持 Y 轴方向不变，以两个已知的连通体的中心点作为X轴，将模型的世界坐标系变换到模型的局部坐标系

    Args:
        mesh (trimesh.Trimesh): 待处理网格
        x_axis (np.array): x 轴的向量

    Returns:
        trimesh.Trimesh: 变换后的网格
    """
    # Y轴：[0, 1, 0]，保持模型咬合面不变
    # 求逆，从而实现世界坐标系变换到局部坐标系
    rotation_matrix = np.matrix(
        matrix_from_two_vectors(x_axis, np.array([0, 1, 0]))
    ).I.__array__()
    rotation_matrix = np.vstack(
        (
            np.hstack((rotation_matrix, np.zeros(3).reshape(-1, 1))),
            np.zeros(4).reshape(1, -1),
        )
    )
    mesh.apply_transform(rotation_matrix)  # 堆叠成四维矩阵才可作为变换矩阵
    return mesh


def normalize_mesh(
    mesh: trimesh.Trimesh,
    ori_mesh_list: [trimesh.Trimesh],  # type: ignore
) -> trimesh.Trimesh:
    """网格均值归一化，并且将网格旋转保证三颗牙的方向是X轴，咬合面的方向不变

    Args:
        mesh (Trimesh): 待处理网格
        ori_mesh([Trimesh]): 预处理前的原始网格

    Returns:
        Trimesh: 均值归一化后的网格，$x\in [-0.5,0.5]$
    """
    # 手工计算OBB，通过两个连通体的中心坐标构成一个向量，与y轴单位向量一起构成一个坐标系，
    # 将原始坐标系线性变换到新的坐标系即可
    prepare_centroid_list = [
        np.array(ori_mesh_list[0].centroid),
        np.array(ori_mesh_list[1].centroid),
    ]
    x_axis = prepare_centroid_list[0] - prepare_centroid_list[1]
    rotation_matrix = np.matrix(
        matrix_from_two_vectors(x_axis, np.array([0, 1, 0]))
    ).I.__array__()
    rotation_matrix = np.vstack(
        (
            np.hstack((rotation_matrix, np.zeros(3).reshape(-1, 1))),
            np.zeros(4).reshape(1, -1),
        )
    )
    ori_mesh = merge_items(ori_mesh_list)
    target_mesh = obb_mesh(ori_mesh, x_axis)

    scale_value = 40
    mesh.apply_scale(scale_value)
    mesh.apply_translation(target_mesh.bounding_box.centroid)
    mesh.apply_transform(np.linalg.pinv(rotation_matrix))
    return mesh


def angle_between_vectors(a, b, ignore=-1):
    """
    计算两个向量之间的夹角（单位：弧度）
    """
    if ignore != -1:
        a[ignore] = b[ignore]
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)
    return angle_rad


def get_roi(mesh, centroid, size):
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [1, 0, 0], centroid[0] - np.array([size, 0, 0])
    )
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [-1, 0, 0], centroid[0] + np.array([size, 0, 0])
    )
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [0, 1, 0], centroid[1] - np.array([0, size, 0])
    )
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [0, -1, 0], centroid[1] + np.array([0, size, 0])
    )
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [0, 0, 1], centroid[2] - np.array([0, 0, size])
    )
    mesh = trimesh.base.intersections.slice_mesh_plane(
        mesh, [0, 0, -1], centroid[2] + np.array([0, 0, size])
    )
    return mesh


def find_new_points(mesh, points, mode=0):
    """
    网格重构后，查找原来的点在新网格中的位置和索引。
    参数:
    mesh (trimesh.Trimesh): 输入的网格。
    points (numpy.ndarray): 网格中原来点的位置。
    mode (int): 选择查找模式。
    如果为0，则查找与原来点距离不超过1e-3的顶点，如果没有则舍弃原来的点。
    如果为1，则查找与原来点距离最近的顶点。
    返回:
    points_ori (numpy.ndarray): 网格中现在点的位置。
    points_id_ori (list): 网格中现在点的ID。
    """
    points_ori = []
    points_id_ori = []
    if mode == 1:
        points = mesh.vertices[
            mesh.faces[trimesh.base.proximity.closest_point(mesh, points)[2]][:, 0]
        ]
    for i in range(len(points)):
        n = np.where(np.linalg.norm(mesh.vertices - points[i], axis=1) < 1e-3)[0]
        if len(n):
            points_id_ori.append(n[0])
            points_ori.append(points[i])
    return points_ori, points_id_ori


def get_neighbors(p_ids, iter_num, mesh):
    def get_next_neighbor(n_id):
        neighbors_id = []
        for i in n_id:
            neighbors_id.extend(mesh.vertex_neighbors[i])
        return list(set(neighbors_id))

    out_ids = []
    for p_id in p_ids:
        out_id = [p_id]
        for _ in range(iter_num):
            out_id.extend(get_next_neighbor(out_id))
        out_ids.append(np.unique(out_id))
    return out_ids


def get_distance(mesh, points_id, cutoff):
    """
    计算网格中指定点到其他点的最短路径距离
    
    参数:
        mesh: 三角网格对象
        points_id: 起始点的索引列表
        cutoff: 距离阈值，超过此距离的点将被忽略
        
    返回:
        在cutoff距离范围内的所有点的索引列表
    """
    # 创建一个无向图
    G = nx.Graph()

    # 将网格的每个顶点添加到图中，并记录其3D坐标
    for v_index, v in enumerate(mesh.vertices):
        G.add_node(v_index, pos=v)

    # 遍历网格的所有边，计算边的权重（即边的长度）
    for edge in mesh.edges_unique:
        v1 = edge[0]  # 边的第一个顶点索引
        v2 = edge[1]  # 边的第二个顶点索引
        # 计算边的权重（两点之间的欧氏距离）
        G.add_edge(v1, v2, weight=np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))

    # 使用Dijkstra算法计算从起始点到其他点的最短路径
    # multi_source_dijkstra_path可以同时处理多个起始点
    paths = nx.multi_source_dijkstra_path(G, points_id, cutoff)
    # 返回所有在cutoff距离范围内的点的索引
    return list(paths.keys())


def find_edge_st_mesh(points, slice_points, mesh, return_angle=False):
    """
    在网格中查找边缘点，并计算相关角度
    
    参数:
        points: 第一组点集
        slice_points: 第二组点集（切片点）
        mesh: 三角网格对象
        return_angle: 是否返回角度信息，默认为False
        
    返回:
        如果return_angle为True，返回边缘点和角度信息
        否则只返回边缘点
    """
    # 计算第一组点的中心点
    c1 = np.mean(np.array(points), axis=0)
    # 获取第二组点集
    c2_p = slice_points
    # 计算第二组点的中心点
    c2 = np.mean(c2_p, axis=0)
    line1 = np.array(points) - c1
    x_z1 = np.arctan2(line1[:, 2], line1[:, 0])
    line2 = c2_p - c2
    x_z2 = np.arctan2(line2[:, 2], line2[:, 0])
    ori_point = []
    point = []
    points_id = []
    points_ = []
    angle = []
    angle_ = []
    # points = np.divide(line1, np.linalg.norm(line1, axis=1).reshape(-1, 1).repeat(3, 1)) * 0.2 + np.array(points)
    for n_p, p in enumerate(x_z1):
        p_id = np.argmin(np.abs(x_z2 - p))
        if np.abs((x_z2 - p)[p_id]) < 0.1:
            ori_point.append(list(points[n_p]))
            point.append(c2_p[p_id])
            angle.append((x_z2 - p)[p_id])
    if mesh:
        for i in range(len(point)):
            n = np.where(mesh.vertices == point[i])[0]
            point_id, num = np.unique(n, return_counts=True)
            num_id = np.where(num == 3)[0]
            if len(num_id):
                points_id.append(point_id[num_id].item())
                points_.append(ori_point[i])
                angle_.append(angle[i])
        if return_angle:
            return points_id, points_, angle_
        else:
            return points_id, points_
    else:
        return point


def find_shortest_path_to_boundary(mesh, points_id):
    """
    计算网格上指定点到边缘的最短路径
    
    参数:
        mesh: 三角网格对象
        points_id: 起始点的索引列表
        
    返回:
        paths: 字典，key为起始点索引，value为到边缘的最短路径点索引列表
        distances: 字典，key为起始点索引，value为到边缘的最短距离
    """
    # 获取网格的所有边界点
    boundaries = find_boundaries(mesh)
    boundary_points = []
    for boundary in boundaries:
        boundary_points.extend(boundary)
    boundary_points = list(set(boundary_points))  # 去重
    
    # 创建图结构
    G = nx.Graph()
    
    # 添加所有顶点
    for v_index, v in enumerate(mesh.vertices):
        G.add_node(v_index, pos=v)
    
    # 添加所有边及其权重
    for edge in mesh.edges_unique:
        v1 = edge[0]
        v2 = edge[1]
        G.add_edge(v1, v2, weight=np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
    
    # 存储结果
    paths = {}
    distances = {}
    
    # 对每个起始点计算到所有边界点的最短路径
    for start_point in points_id:
        min_distance = float('inf')
        min_path = None
        
        # 计算到每个边界点的最短路径
        for boundary_point in boundary_points:
            try:
                path = nx.shortest_path(G, source=start_point, target=boundary_point, weight='weight')
                distance = nx.shortest_path_length(G, source=start_point, target=boundary_point, weight='weight')
                
                # 更新最短路径
                if distance < min_distance:
                    min_distance = distance
                    min_path = path
            except nx.NetworkXNoPath:
                continue
        
        # 存储结果
        if min_path is not None:
            paths[start_point] = min_path
            distances[start_point] = min_distance
    
    return paths, distances


def compute_signed_distance(mesh, q_points):
    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)
    closest_points = scene.compute_closest_points(
        np.asarray(q_points, dtype=np.float32)
    )

    distance = np.linalg.norm(q_points - closest_points["points"].numpy(), axis=-1)
    nonzero = distance > 0

    # For closest points that project directly in to the triangle, compute sign from
    # triangle normal Project each point in to the closest triangle plane
    nonzero = np.where(nonzero)[0]
    normals = closest_points["primitive_normals"].numpy()
    projection = (
        q_points[nonzero]
        - (
            normals[nonzero].T
            * np.einsum(
                "ij,ij->i",
                q_points[nonzero] - closest_points["points"].numpy()[nonzero],
                normals[nonzero],
            )
        ).T
    )

    sign = np.sign(
        np.einsum("ij,ij->i", normals[nonzero], q_points[nonzero] - projection)
    )
    distance[nonzero] *= -1.0 * sign
    return distance, closest_points["points"].numpy()


def find_boundaries(mesh):
    if type(mesh) is trimesh.Trimesh or type(mesh) is TrackedTrimesh:
        mesh_o3d = mesh.as_open3d
    else:
        mesh_o3d = mesh
    a = mesh_o3d.get_non_manifold_edges(allow_boundary_edges=True)
    b = mesh_o3d.get_non_manifold_edges(allow_boundary_edges=False)
    a = np.unique(np.asarray(a).flatten())
    b = np.unique(np.asarray(b).flatten())
    out = b[~np.isin(b, a)]

    return np.asarray(mesh.vertices)[out], out
    # out = np.asarray(b)[~np.isin(np.asarray(b)[:,0], np.asarray(a)[:,0]) | ~np.isin(np.asarray(b)[:,1], np.asarray(a)[:,1])]
    # return mesh.vertices[out.flatten()], out.flatten()


seed = 2023
random.seed(seed)
np.random.seed(seed)


class GenerateCrowns:
    def __init__(self):
        self.minimal_thickness = 0.6
        self.margin_width = 0.2
        self.cement_gap = 0.12
        self.height_of_minimal_gap = 0.8
        self.height_of_minimal_gap2 = 0.5
        self.minimal_gap = 0.04
        self.occlusal_distance = 0.3
        self.ad_gap = -0.03
        self.prox_or_occlu = 2
        self.adjust_crown = 1
        self.dac_point = None
        self.mesh_upper = None
        self.mesh_lower = None
        self.mesh_oppo = None
        self.ori_mesh = None
        self.mesh_outside_without_thickness = None
        self.mesh_without_thickness = None
        self.transform = None
        self.new_transform_list = None
        self.paras = None
        self.AB = None
        self.test = False
        self.test_path = None
        self.points_ori = None
        self.add_points = None
        self.add_point_id = None
        self.adj_points_id1 = None
        self.adj_points_id2 = None
        self.cross_points = None
        self.cross_point_id = None
        self.linya_points = None
        self.filled = False
        self.align_edges = True
        self.stitch_success = True
        self.axis = None
        self.thick_flag = 1
        self.points_info = None
        self.morph_template = None
        self.prep_extended = None
        self.template_name = "st_tooth"
        self.AOI_or_UB = 1

    def load_data(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def get_adjacent_area(self):
        p1 = self.mesh.cross_points.pt[2]
        p2 = self.mesh.cross_points.pt[3]

        if np.dot((p1 - p2), self.axis_y) > 0:
            max_edge = p1 + self.axis_y / np.linalg.norm(self.axis_y) * 0.5
            min_edge = p2 - self.axis_y / np.linalg.norm(self.axis_y) * 0.5
        else:
            max_edge = p2 + self.axis_y / np.linalg.norm(self.axis_y) * 0.5
            min_edge = p1 - self.axis_y / np.linalg.norm(self.axis_y) * 0.5
        plan1 = (max_edge, self.axis_y * -1)
        plan2 = (min_edge, self.axis_y)
        mesh_slice = self.mesh.slice_plane(plan1[0], plan1[1])
        mesh_slice = mesh_slice.slice_plane(plan2[0], plan2[1])
        adj_points = mesh_slice.vertices
        max_x = max(adj_points[:, 0])
        min_x = min(adj_points[:, 0])
        adj_points1 = adj_points[
            np.where(adj_points[:, 0] > (max_x - (max_x - min_x) / 5))[0]
        ]
        adj_points2 = adj_points[
            np.where(adj_points[:, 0] < (min_x + (max_x - min_x) / 5))[0]
        ]
        _, adj_points_id1 = find_new_points(self.mesh, adj_points1, 1)
        _, adj_points_id2 = find_new_points(self.mesh, adj_points2, 1)
        self.mesh.add_keypoint("adj_points1", adj_points_id1)
        self.mesh.add_keypoint("adj_points2", adj_points_id2)

    def read_mesh(self):
        self.mesh_beiya = read_mesh_bytes(self.mesh_beiya)
        self.mesh_beiya = get_biggest_mesh(self.mesh_beiya)
        if self.mesh_upper:
            self.mesh_upper = read_mesh_bytes(self.mesh_upper)
            if self.transform:
                self.mesh_upper.apply_transform(self.transform)
        if self.mesh_lower:
            self.mesh_lower = read_mesh_bytes(self.mesh_lower)
            if self.transform:
                self.mesh_lower.apply_transform(self.transform)
        if self.miss_id in ["17", "27", "37", "47"]:
            self.is_single = 2
        if int(self.miss_id) < 30:
            self.up_low = 0
        else:
            self.up_low = 1

        self.load_config()

        if self.is_single == 1:
            self.use_pt1 = False
            self.use_pt2 = self.check_kps(self.pt2, int(self.miss_id) + 1)
        elif self.is_single == 2:
            if self.miss_id in ["14", "24", "34", "44"]:
                self.use_pt1 = True
            else:
                self.use_pt1 = self.check_kps(self.pt1, int(self.miss_id) - 1)
            self.use_pt2 = False
        elif self.is_single == 0:
            if self.miss_id in ["14", "24", "34", "44"]:
                self.use_pt1 = True
            else:
                self.use_pt1 = self.check_kps(self.pt1, int(self.miss_id) - 1)
            self.use_pt2 = self.check_kps(self.pt2, int(self.miss_id) + 1)
        if self.is_single != 3:
            self.get_dac_point()
            self.get_linya()
            # self.get_crown()

    def load_config(self):
        self.cwd = os.getcwd()
        # 读配置文件
        with open(
            os.path.join(self.cwd, "config", self.template_name, "config.json"), "r"
        ) as j:
            config = json.load(j)
        standard_mesh = os.path.join(self.cwd, config[self.miss_id]["st_path"])
        standard_mesh = trimesh.load(standard_mesh)
        self.mesh = TrackedTrimesh(standard_mesh.vertices, standard_mesh.faces)
        for key_name in list(config[self.miss_id].keys()):
            if key_name not in ["st_path", "pt_path", "points_oppo_id"]:
                self.mesh.add_keypoint(key_name, config[self.miss_id][key_name])
        self.points_oppo_id = config[self.miss_id]["points_oppo_id"]

        self.z_length = config["z_length"]

    def load_points(self):
        self.mesh = TrackedTrimesh(self.mesh.vertices, self.mesh.faces)
        if self.trans_matrix is None:
            self.trans_matrix = np.eye(4)
        elif np.array(self.trans_matrix).ndim == 1:
            self.trans_matrix = np.array(self.trans_matrix).reshape((4, 4))
        if self.points_info is not None and self.morph_template == self.template_name:
            p_id = {}
            # str2color = {}
            # color_list = [
            #     x for x in [x for y in list(self.cpu_colors_info.values()) for x in y]
            # ]
            # for color in color_list:
            #     str2color["".join(map(str, color))] = color
            for color_str in self.points_info.keys():
                # color = str2color[color_str]
                color = [int(x) for x in color_str.split('-')]
                p_id[
                    bin(color[0])[2:].zfill(8)
                    + bin(color[1])[2:].zfill(8)
                    + bin(color[2])[2:].zfill(8)
                ] = color_str
            for idx, (k, v) in enumerate(self.cpu_colors_info.items()):
                pc = []
                for s, c in p_id.items():
                    if s[idx] == "1":
                        if k in [
                            "ad_points",
                        ]:
                            pc.append((self.points_info[c][0], int(s[-8:-4], 2)))
                        elif k in [
                            "cross_points",
                        ]:
                            pc.append((self.points_info[c][0], int(s[-4:], 2)))
                        else:
                            pc.extend(self.points_info[c])
                if k in [
                    "ad_points",
                    "cross_points",
                ]:
                    pc = [x[0] for x in sorted(pc, key=lambda x: x[1])]
                self.mesh.add_keypoint(k, pt=np.array(pc), mode=1)
        else:
            for k, v in self.cpu_points_info.items():
                if self.handler_name == "post":
                    pc = trimesh.PointCloud(v)
                    pc.apply_transform(self.trans_matrix)
                    self.mesh.add_keypoint(k, pt=pc.vertices, mode=1)
                elif self.handler_name == "occlu":
                    if k in [
                        "st_points",
                        "ad_points",
                        "cross_points",
                        "adj_points1",
                        "adj_points2",
                        "linya_points",
                        "occl_points",
                    ] and v:
                        self.mesh.add_keypoint(k, pt=np.array(v), mode=1)
        if self.axis:
            axis_ply = trimesh.PointCloud(self.axis)
            axis_ply.apply_transform(self.trans_matrix)
            self.axis = np.array(axis_ply.vertices)
            self.print_matrix = rotate_from_axis(self.axis)

    def load_paras(self):
        if self.paras:
            for key, value in self.paras.items():
                if value is not None:
                    self.__setattr__(key, value)

    def get_dac_point(self):
        self.dac_point = None
        if self.is_single in [1, 2] and len(self.all_other_crowns) > 9:
            dac = DAC(
                self.all_other_crowns, self.kps, int(self.miss_id), self.is_single
            )
            curve, control_points, sampled_points, T, A = dac.get_dac_nurbs()
            if self.is_single == 1:
                crown_id_1 = int(self.miss_id) - 1
                crown_id_2 = int(self.miss_id) + 1
            elif self.is_single == 2:
                crown_id_1 = int(self.miss_id) + 1
                crown_id_2 = int(self.miss_id) - 1
            if crown_id_1 < 20 or 30 < crown_id_1 < 40:
                point_num_1 = 8 - (crown_id_1 % 10)
            else:
                point_num_1 = crown_id_1 % 10 + 7
            if crown_id_2 < 20 or 30 < crown_id_2 < 40:
                point_num_2 = 8 - (crown_id_2 % 10)
            else:
                point_num_2 = crown_id_2 % 10 + 7

            # vis
            mesh = (
                self.mesh_lower.copy()
                if dac.up_or_low == "lower"
                else self.mesh_upper.copy()
            )
            mesh.apply_transform(T)

            control_points = trimesh.PointCloud(control_points)
            control_points.apply_transform(np.linalg.pinv(T))
            self.dac_point = [control_points[point_num_1], control_points[point_num_2]]

    def get_linya(self):
        centroid = self.mesh_beiya.centroid
        if self.is_single == 1:
            self.mesh2 = read_mesh_bytes(self.mesh2)
            self.mesh1 = self.mesh2.copy()
            self.mesh1.apply_translation(-(self.mesh2.centroid - centroid) * 2)
            if self.dac_point:
                trans_h = self.dac_point[0][1] - self.mesh1.centroid[1]
                self.mesh1.apply_translation([0, trans_h, 0])
            if self.pt2 and self.use_pt2:
                self.pt2 = trimesh.PointCloud(self.pt2)
                self.pt1 = self.pt2.copy()
                self.pt1.apply_translation(-(self.mesh2.centroid - centroid) * 2)
                if self.dac_point:
                    self.pt1.apply_translation([0, trans_h, 0])
            else:
                self.pt1 = None
                self.pt2 = None
        elif self.is_single == 2:
            self.mesh1 = read_mesh_bytes(self.mesh1)
            self.mesh2 = self.mesh1.copy()
            self.mesh2.apply_translation(-(self.mesh1.centroid - centroid) * 2)
            if self.dac_point:
                trans_h = self.dac_point[0][1] - self.mesh2.centroid[1]
                self.mesh2.apply_translation([0, trans_h, 0])
            if self.pt1 and self.use_pt1:
                self.pt1 = trimesh.PointCloud(self.pt1)
                self.pt2 = self.pt1.copy()
                self.pt2.apply_translation(-(self.mesh1.centroid - centroid) * 2)
                if self.dac_point:
                    self.pt2.apply_translation([0, trans_h, 0])
            else:
                self.pt1 = None
                self.pt2 = None
        else:
            self.mesh1 = read_mesh_bytes(self.mesh1)
            self.mesh2 = read_mesh_bytes(self.mesh2)
            if self.pt1 and self.use_pt1:
                self.pt1 = trimesh.PointCloud(self.pt1)
                if self.pt2 and self.use_pt2:
                    self.pt2 = trimesh.PointCloud(self.pt2)
                else:
                    self.pt2 = self.pt1.copy()
                    self.pt2.apply_translation(-(self.mesh1.centroid - centroid) * 2)
            elif self.pt2 and self.use_pt2:
                self.pt2 = trimesh.PointCloud(self.pt2)
                self.pt1 = self.pt2.copy()
                self.pt1.apply_translation(-(self.mesh2.centroid - centroid) * 2)
            else:
                self.pt1 = None
                self.pt2 = None

    def get_crown(self):
        self.voxel_logits = np.array(self.voxel_logits)
        pc = trimesh.points.PointCloud(self.voxel_logits)
        pc.apply_scale(1 / 256)
        pc.apply_translation([-0.5, -0.5, -0.5])
        pc = normalize_mesh(
            pc,
            [
                self.mesh1.copy(),
                self.mesh2.copy(),
            ],
        )

        matrix, _, _ = trimesh.base.registration.procrustes(
            self.mesh.st_points.pt,
            pc.vertices,
            reflection=False,
        )
        self.mesh.apply_transform(matrix)
        self.mesh.apply_translation(
            self.mesh_beiya.bounding_box_oriented.centroid
            - self.mesh.bounding_box_oriented.centroid
        )
        self.mesh.update_mesh()

    def get_matrix_from_mesh(self):
        if self.new_transform_list:
            self.rotation_matrix = np.array(self.new_transform_list[0])
            self.matrix = np.array(self.new_transform_list[1])
        else:
            if int(self.miss_id) in [17, 27, 37, 47] and self.dac_point is not None:
                crown_axis = read_mesh_bytes(
                    self.all_other_crowns[str(int(self.miss_id) - 2)]
                )
                axis_x = crown_axis.centroid - self.mesh1.centroid
            else:
                axis_x = self.mesh1.centroid - self.mesh2.centroid
            # if self.up_low:
            #     self.rotation_matrix = np.matrix(
            #         matrix_from_two_vectors(axis_x, np.array([0, 1, 0]))
            #     ).I.__array__()
            # else:
            #     self.rotation_matrix = np.matrix(
            #         matrix_from_two_vectors(axis_x, np.array([0, -1, 0]))
            #     ).I.__array__()
            self.rotation_matrix = np.matrix(
                matrix_from_two_vectors(axis_x, np.array(self.axis_y))
            ).I.__array__()
            self.rotation_matrix = np.vstack(
                (
                    np.hstack((self.rotation_matrix, np.zeros(3).reshape(-1, 1))),
                    np.zeros(4).reshape(1, -1),
                )
            )
            self.matrix = np.eye(4)
            self.matrix[:3, 3] = self.mesh.centroid * -1
        self.mesh.apply_transform(self.rotation_matrix @ self.matrix)
        self.mesh_beiya.apply_transform(self.rotation_matrix @ self.matrix)
        self.mesh1.apply_transform(self.rotation_matrix @ self.matrix)
        self.mesh2.apply_transform(self.rotation_matrix @ self.matrix)
        if self.mesh_upper:
            self.mesh_upper.apply_transform(self.rotation_matrix @ self.matrix)
        if self.mesh_lower:
            self.mesh_lower.apply_transform(self.rotation_matrix @ self.matrix)
        if self.pt1:
            self.pt1.apply_transform(self.rotation_matrix @ self.matrix)
        if self.pt2:
            self.pt2.apply_transform(self.rotation_matrix @ self.matrix)
        if self.AB:
            self.AB.apply_transform(self.rotation_matrix @ self.matrix)

    def get_matrix_from_pt(self):
        if self.new_transform_list:
            self.pt_matrix = np.array(self.new_transform_list[2])
        else:
            if self.pt1 and self.pt2:
                y1 = self.pt1.vertices[0] - np.mean(
                    [self.pt1.vertices[1], self.pt1.vertices[2]], axis=0
                )
                y2 = self.pt2.vertices[0] - np.mean(
                    [self.pt2.vertices[1], self.pt2.vertices[2]], axis=0
                )
                if (y1 + y2)[2] < 0:
                    m = trimesh.base.transformations.quaternion_about_axis(
                        np.arccos(np.dot(y1 + y2, [0, 1, 0]) / np.linalg.norm(y1 + y2)),
                        [1, 0, 0],
                    )
                else:
                    m = trimesh.base.transformations.quaternion_about_axis(
                        -np.arccos(
                            np.dot(y1 + y2, [0, 1, 0]) / np.linalg.norm(y1 + y2)
                        ),
                        [1, 0, 0],
                    )
                self.pt_matrix = trimesh.base.transformations.quaternion_matrix(m)

            else:
                self.pt_matrix = []
        if len(self.pt_matrix):
            self.mesh_beiya.apply_transform(self.pt_matrix)
            self.mesh1.apply_transform(self.pt_matrix)
            self.mesh2.apply_transform(self.pt_matrix)
            if self.mesh_upper:
                self.mesh_upper.apply_transform(self.pt_matrix)
            if self.mesh_lower:
                self.mesh_lower.apply_transform(self.pt_matrix)
            if self.pt1:
                self.pt1.apply_transform(self.pt_matrix)
            if self.pt2:
                self.pt2.apply_transform(self.pt_matrix)

    def get_matrix_from_ab(self):
        unit_vector = self.AB.vertices[0] - self.AB.vertices[1]
        if unit_vector[2] < 0:
            m = trimesh.base.transformations.quaternion_about_axis(
                np.arccos(np.dot(unit_vector, [0, 1, 0]) / np.linalg.norm(unit_vector)),
                [1, 0, 0],
            )
        else:
            m = trimesh.base.transformations.quaternion_about_axis(
                -np.arccos(
                    np.dot(unit_vector, [0, 1, 0]) / np.linalg.norm(unit_vector)
                ),
                [1, 0, 0],
            )
        self.pt_matrix = trimesh.base.transformations.quaternion_matrix(m)
        self.mesh_beiya.apply_transform(self.pt_matrix)
        self.mesh1.apply_transform(self.pt_matrix)
        self.mesh2.apply_transform(self.pt_matrix)
        if self.mesh_upper:
            self.mesh_upper.apply_transform(self.pt_matrix)
        if self.mesh_lower:
            self.mesh_lower.apply_transform(self.pt_matrix)
        self.pt1.apply_transform(self.pt_matrix)
        self.pt2.apply_transform(self.pt_matrix)
        self.AB.apply_transform(self.pt_matrix)

    def get_matrix_from_ai(self):
        self.mesh_beiya.apply_transform(self.ai_matrix)
        if self.mesh_upper:
            self.mesh_upper.apply_transform(self.ai_matrix)
        if self.mesh_lower:
            self.mesh_lower.apply_transform(self.ai_matrix)
        self.mesh1.apply_transform(self.ai_matrix)
        self.mesh2.apply_transform(self.ai_matrix)

        if self.pt1:
            self.pt1.apply_transform(self.ai_matrix)
        if self.pt2:
            self.pt2.apply_transform(self.ai_matrix)
        if self.AB:
            self.AB.apply_transform(self.ai_matrix)

        self.mesh.update_mesh()

        v_occl = self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1]
        v_misial = self.mesh.cross_points.pt[2] - self.mesh.cross_points.pt[3]
        frameA = np.array([v_occl, v_misial])
        frameB = np.array([[0, 1, 0], [1, 0, 0]])
        weights = np.array([0.5, 0.5])
        rot_mat, root_sum_squared_distance = transform.Rotation.align_vectors(
            frameA, frameB, weights=weights
        )
        rot_matrix = np.eye(4)  # 创建一个单位矩阵作为变换矩阵的初始值
        rot_matrix[:3, :3] = (
            rot_mat.as_matrix()
        )  # 复制旋转矩阵的前三列到变换矩阵的前三列
        rot_matrix[:, 3] = [0, 0, 0, 1]
        self.mesh.apply_transform(np.linalg.pinv(rot_matrix))
        self.mesh.update_mesh()

    def get_roi_from_upper_lower(self):
        if self.up_low:
            self.mesh_jaw = self.mesh_lower.copy()
            if self.mesh_upper:
                self.mesh_oppo = self.mesh_upper.copy()
        else:
            self.mesh_jaw = self.mesh_upper.copy()
            if self.mesh_lower:
                self.mesh_oppo = self.mesh_lower.copy()
        self.mesh_jaw = get_roi(self.mesh_jaw, self.mesh_beiya.centroid, 10)
        self.mesh_jaw = self.mesh_jaw.simplify_quadric_decimation(25000)
        if self.mesh_oppo:
            self.mesh_oppo = get_roi(self.mesh_oppo, self.mesh_beiya.centroid, 20)
        self.mesh_jaw = get_biggest_mesh(self.mesh_jaw)

    def get_purue_beiya(self):
        self.mesh_jaw = self.mesh_jaw.as_open3d
        self.mesh_jaw.remove_non_manifold_edges()
        while True:
            non_mani_p = self.mesh_jaw.get_non_manifold_vertices()
            if len(non_mani_p):
                self.mesh_jaw.remove_vertices_by_index(non_mani_p)
            else:
                break
        self.mesh_jaw = trimesh.Trimesh(self.mesh_jaw.vertices, self.mesh_jaw.triangles)

        interest_verts = self.mesh_jaw.nearest.vertex(self.mesh_beiya.vertices)[1]

        self.mesh_beiya = pypruners.pruner(
            self.mesh_jaw.vertices, self.mesh_jaw.faces, interest_verts, 1.0, -2.5, 1.5
        )
        self.neck_points, _ = find_boundaries(self.mesh_beiya)

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

    def get_cement_gap(self):
        """
        备牙膨胀，用于留出填充间隙。
        首先修复网格的法线，然后遍历网格的每个顶点。
        对于每个顶点，计算其法线和相邻顶点的法线，如果角度大于60度，不对该点做调整。
        反之，使用法线移动该顶点。
        处理所有顶点后，使用TPS对网格进行重采样。

        """
        self.mesh_beiya.fix_normals()
        points = []
        outlines = self.mesh_beiya.outline().referenced_vertices
        # 构建 KD 树
        tree = KDTree(self.mesh_beiya.vertices[outlines])
        # 计算每个点到另一个点云的最小距离
        distances, _ = tree.query(self.mesh_beiya.vertices)
        n = []
        for k in range(len(self.mesh_beiya.vertices)):
            if k in outlines:
                points.append(self.mesh_beiya.vertices[k])
                continue
            a = self.mesh_beiya.vertex_normals[k]
            b = self.mesh_beiya.vertex_normals[self.mesh_beiya.vertex_neighbors[k]]
            cos_angle = [np.dot(a, x) for x in b]
            angle = np.min(np.arccos(cos_angle) / 3.141592653 * 180)
            if angle > 60:
                n.append(k)
                points.append(self.mesh_beiya.vertices[k])
                continue
            else:
                v_n = a
            points.append(self.mesh_beiya.vertices[k] + v_n * self.cement(distances[k]))
        points_id = random.sample(
            [x for x in range(len(self.mesh_beiya.vertices))],
            len(self.mesh_beiya.vertices) // 4,
        )
        points_id = [x for x in points_id if x not in n]
        points_tps = np.array(points)[points_id]
        self.mesh_beiya = tps(self.mesh_beiya, points_id, points_tps)

    def dilation(self):
        # self.mesh_beiya = self.mesh_beiya.as_open3d
        # self.mesh_beiya.merge_close_vertices(0.05)
        # self.mesh_beiya = trimesh.Trimesh(
        #     self.mesh_beiya.vertices, self.mesh_beiya.triangles
        # )
        if len(self.mesh_beiya.faces) > 10000:
            self.mesh_beiya = self.mesh_beiya.simplify_quadric_decimation(10000)
        self.mesh_beiya = get_biggest_mesh(self.mesh_beiya)
        self.get_cement_gap()


    def trans_scale(self):
        mesh_centroid = (self.mesh_beiya.centroid).copy()
        self.mesh.apply_translation(mesh_centroid - self.mesh.centroid)
        self.mesh.update_mesh(self.mesh.vertices)
        self.neck_points, _ = find_boundaries(self.mesh_beiya)
        self.mesh1_id = str(int(self.miss_id) - 1)
        self.mesh2_id = str(int(self.miss_id) + 1)

        if self.is_single == 1:
            self.mesh1_id = str(int(self.miss_id) - 2)
            if self.mesh1_id in self.all_other_crowns.keys():
                mesh1_ = read_mesh_bytes(self.all_other_crowns[self.mesh1_id])
                if mesh1_.bounds[0, 0] - self.mesh.bounds[1, 0] < 2:
                    self.is_single = 0
                    self.mesh1 = mesh1_
                    self.mesh1.apply_transform(self.rotation_matrix @ self.matrix)
                    if len(self.pt_matrix):
                        self.mesh1.apply_transform(self.pt_matrix)
                    self.mesh1.apply_transform(self.ai_matrix)
                    if (
                        self.mesh1_id
                        in list(list(self.kps.values())[0].values())[0].keys()
                    ):
                        self.pt1 = trimesh.PointCloud(
                            list(
                                list(list(self.kps.values())[0].values())[0][
                                    self.mesh1_id
                                ].values()
                            )
                        )
                        self.pt1.apply_transform(self.rotation_matrix @ self.matrix)
                        if len(self.pt_matrix):
                            self.pt1.apply_transform(self.pt_matrix)
                        self.pt1.apply_transform(self.ai_matrix)
                        if self.mesh1_id[1] == "3":
                            self.use_pt1 = False
                        else:
                            self.use_pt1 = self.check_kps(
                                self.pt1.vertices.tolist(), self.mesh1_id
                            )
                    else:
                        self.use_pt1 = False
                        self.pt1 = None
        elif self.is_single == 2:
            self.mesh2_id = str(int(self.miss_id) + 2)
            if self.mesh2_id in self.all_other_crowns.keys():
                mesh2_ = read_mesh_bytes(self.all_other_crowns[self.mesh2_id])
                if self.mesh.bounds[0, 0] - mesh2_.bounds[1, 0] < 2:
                    self.is_single = 0
                    self.mesh2 = mesh2_
                    self.mesh2.apply_transform(self.rotation_matrix @ self.matrix)
                    if len(self.pt_matrix):
                        self.mesh2.apply_transform(self.pt_matrix)
                    self.mesh2.apply_transform(self.ai_matrix)
                    if (
                        self.mesh2_id
                        in list(list(self.kps.values())[0].values())[0].keys()
                    ):
                        self.pt2 = trimesh.PointCloud(
                            list(
                                list(list(self.kps.values())[0].values())[0][
                                    self.mesh2_id
                                ].values()
                            )
                        )
                        self.pt2.apply_transform(self.rotation_matrix @ self.matrix)
                        if len(self.pt_matrix):
                            self.pt2.apply_transform(self.pt_matrix)
                        self.pt2.apply_transform(self.ai_matrix)
                        self.use_pt2 = self.check_kps(
                            self.pt2.vertices.tolist(), self.mesh2_id
                        )
                    else:
                        self.use_pt2 = False
                        self.pt2 = None

        path_x1 = trimesh.intersections.mesh_plane(
            self.mesh1, [0, 0, 1], [0, 0, 0]
        ).reshape(-1, 3)[:, 0]
        path_x2 = trimesh.intersections.mesh_plane(
            self.mesh2, [0, 0, 1], [0, 0, 0]
        ).reshape(-1, 3)[:, 0]
        path_x = trimesh.intersections.mesh_plane(
            self.mesh, [0, 0, 1], [0, 0, 0]
        ).reshape(-1, 3)[:, 0]

        if self.is_single == 1:
            max_x = np.max(self.neck_points[:, 0])
            min_x = np.max(path_x2)
        elif self.is_single == 2:
            max_x = np.min(path_x1)
            min_x = np.min(self.neck_points[:, 0])
        elif self.is_single == 0:
            max_x = np.min(path_x1)
            min_x = np.max(path_x2)
        dis_x = np.max(path_x) - np.min(path_x)
        dis_z_1 = np.diff(self.mesh1.bounds, axis=0)[0, 2]
        dis_z_2 = np.diff(self.mesh2.bounds, axis=0)[0, 2]
        dis_z = np.diff(self.mesh.bounds, axis=0)[0, 2]
        if self.is_single == 1:
            z_length_2 = self.z_length[self.mesh2_id]
            z_length_1 = z_length_2
        elif self.is_single == 2:
            z_length_1 = self.z_length[self.mesh1_id]
            z_length_2 = z_length_1
        elif self.is_single == 0:
            z_length_1 = self.z_length[self.mesh1_id]
            z_length_2 = self.z_length[self.mesh2_id]
        z_length = self.z_length[self.miss_id]

        scale_x = (max_x - min_x) / dis_x
        scale_z = (
            (dis_z_1 / z_length_1 * z_length + dis_z_2 / z_length_2 * z_length)
            / 2
            / dis_z
        )
        self.mesh.apply_scale([scale_x, 1, scale_z])
        self.mesh.update_mesh(self.mesh.vertices)

    def check_kps(self, kps, t_id):
        with open(
            os.path.join(self.cwd, "config", "points", "{}.json".format(t_id)), "r"
        ) as f:
            info = eval(f.read())
        info = [list(i.values()) for i in info]
        kps_list = kps
        kps = [kps_list[0]] + kps_list[3:-4]

        cost_min = 0
        # 遍历50个正常案例，找到与测试案例误差最小的值
        for i in info:
            (
                matrix,
                transformed,
                cost,
            ) = trimesh.base.registration.procrustes(kps, i, reflection=False)
            if cost > cost_min:
                cost_min = cost
        if cost_min > 1:
            return False
        else:
            return True

    def trans_p(self):
        """
        根据邻牙的关键点移动关键点，使其大致在一条线上，如尖点等
        """

        new_pt = []
        new_pt_id = []
        p_z = []
        p_z_p1 = []
        p_z_p2 = []
        p_z_id1 = []
        p_z_id2 = []

        def get_p(p_id, pt1_id, pt2_id, type):
            if type == 0:
                new_p = trimesh.base.intersections.plane_lines(
                    np.array(self.mesh.ad_points.pt)[p_id],
                    self.pt1.vertices[pt1_id] - self.pt2.vertices[pt2_id],
                    np.array(
                        [
                            self.pt1.vertices[pt1_id] * 2 - self.pt2.vertices[pt2_id],
                            self.pt2.vertices[pt2_id] * 2 - self.pt1.vertices[pt1_id],
                        ]
                    ).reshape((2, 1, 3)),
                )[0][0]
                new_pt.append(new_p)
                new_pt_id.append(self.mesh.ad_points.idx[p_id])
            elif type == 1:
                new_p = trimesh.base.intersections.plane_lines(
                    np.array(self.mesh.ad_points.pt)[p_id],
                    self.pt1.vertices[pt1_id] - self.pt1.vertices[pt1_id] * [-1, 1, 1],
                    np.array(
                        [
                            self.pt1.vertices[pt1_id] * 2
                            - self.pt1.vertices[pt1_id] * [-1, 1, 1],
                            self.pt1.vertices[pt1_id] * [-1, 1, 1] * 2
                            - self.pt1.vertices[pt1_id],
                        ]
                    ).reshape((2, 1, 3)),
                )[0][0]
                new_pt.append(new_p)
                new_pt_id.append(self.mesh.ad_points.idx[p_id])
            elif type == 2:
                new_p = trimesh.base.intersections.plane_lines(
                    np.array(self.mesh.ad_points.pt)[p_id],
                    self.pt2.vertices[pt2_id] - self.pt2.vertices[pt2_id] * [-1, 1, 1],
                    np.array(
                        [
                            self.pt2.vertices[pt2_id] * 2
                            - self.pt2.vertices[pt2_id] * [-1, 1, 1],
                            self.pt2.vertices[pt2_id] * [-1, 1, 1] * 2
                            - self.pt2.vertices[pt2_id],
                        ]
                    ).reshape((2, 1, 3)),
                )[0][0]
                new_pt.append(new_p)
                new_pt_id.append(self.mesh.ad_points.idx[p_id])
            else:
                pass

        if int(self.miss_id) in [14, 24, 34, 44]:
            if self.use_pt2:
                get_p(1, -1, 3, 2)
                get_p(2, -1, 4, 2)
                get_p(7, -1, 7, 2)
                get_p(8, -1, 7, 2)
        elif int(self.miss_id) in [15, 25, 35, 45]:
            if self.use_pt1 and self.use_pt2:
                get_p(1, 3, 3, 0)
                get_p(2, 4, 5, 0)
                get_p(7, 8, 7, 0)
                get_p(8, 8, 7, 0)
            elif self.use_pt2:
                get_p(1, -1, 3, 2)
                get_p(2, -1, 5, 2)
                get_p(7, -1, 7, 2)
                get_p(8, -1, 7, 2)
            elif self.use_pt1:
                get_p(1, 3, -1, 1)
                get_p(2, 4, -1, 1)
                get_p(7, 8, -1, 1)
                get_p(8, 8, -1, 1)
        elif int(self.miss_id) in [16, 26, 36, 46]:
            if self.use_pt1 and self.use_pt2:
                get_p(0, 3, 3, 0)
                get_p(1, 3, 3, 0)
                get_p(2, 4, 5, 0)
                get_p(3, 4, 5, 0)
                get_p(7, 8, 7, 0)
                get_p(8, 8, 7, 0)
            elif self.use_pt2:
                get_p(0, -1, 3, 2)
                get_p(1, -1, 3, 2)
                get_p(2, -1, 5, 2)
                get_p(3, -1, 5, 2)
                get_p(7, -1, 7, 2)
                get_p(8, -1, 7, 2)
            elif self.use_pt1:
                get_p(0, 3, -1, 1)
                get_p(1, 3, -1, 1)
                get_p(2, 4, -1, 1)
                get_p(3, 4, -1, 1)
                get_p(7, 8, -1, 1)
                get_p(8, 8, -1, 1)
        elif int(self.miss_id) in [17, 27, 37, 47]:
            if self.use_pt1:
                get_p(0, 4, -1, 1)
                get_p(1, 4, -1, 1)
                get_p(2, 6, -1, 1)
                get_p(3, 6, -1, 1)
                get_p(7, 8, -1, 1)
                get_p(8, 8, -1, 1)
        if new_pt:
            ori_points = self.mesh.vertices[new_pt_id].tolist()
            points = new_pt
            points = np.array(points)
            ori_points = np.array(ori_points)
            mean_ori_points = np.mean(ori_points[[-1, -2], 1])
            mean_points = np.mean(points[[-1, -2], 1])
            points[:, 1] = ori_points[:, 1] + mean_points - mean_ori_points

            points[:, 0] = ori_points[:, 0]
            points[:, 2] = ori_points[:, 2]

            m, _, _ = trimesh.base.registration.procrustes(
                ori_points, points, reflection=False
            )
            self.mesh.apply_transform(m)
            self.mesh.update_mesh()

        p_z = self.mesh.cross_points.pt[-2:].copy()
        self.mesh.cross_points.idx[-2:]

        if self.miss_id[0] in ["1", "3"]:
            dis_p1 = p_z[0][2] - (self.mesh_beiya.bounds[1, 2])
            dis_p2 = self.mesh_beiya.bounds[0, 2] - 0.5 - p_z[1][2]
        elif self.miss_id[0] in ["2", "4"]:
            dis_p1 = p_z[0][2] - (self.mesh_beiya.bounds[0, 2])
            dis_p2 = self.mesh_beiya.bounds[1, 2] + 0.5 - p_z[1][2]
        p_z[0][2] -= dis_p1
        p_z[1][2] += dis_p2
        p_z_id1 = get_distance(self.mesh, [self.mesh.cross_points.idx[-2]], 1)
        p_z_id2 = get_distance(self.mesh, [self.mesh.cross_points.idx[-1]], 1)
        p_z_p1 = self.mesh.vertices[p_z_id1]
        p_z_p2 = self.mesh.vertices[p_z_id2]
        p_z_p1[:, 2] -= dis_p1
        p_z_p2[:, 2] += dis_p2

        points = []
        points_id = []
        points.extend(p_z_p1)
        points.extend(p_z_p2)
        points_id.extend(p_z_id1)
        points_id.extend(p_z_id2)
        self.mesh = tps(self.mesh, points_id, points)
        self.mesh_backup = TrackedTrimesh(self.mesh.vertices, self.mesh.faces)
        # for v in vars(self.mesh):
        #     if v not in vars(trimesh.Trimesh()) and v not in ['keypoints', 'vert_num']:
        #         self.mesh_backup.add_keypoint(v, getattr(self.mesh, v).idx)

        # points = []
        # points_id = []
        # if new_pt:
        #     if self.is_single == 0:
        #         self.mesh.ad_points.pt[7, 1] = self.pt1.vertices[8, 1]
        #         self.mesh.ad_points.pt[8, 1] = self.pt2.vertices[7, 1]
        #     if self.is_single == 1 or int(self.mesh1_id) in [13, 23, 33, 43]:
        #         self.mesh.ad_points.pt[8, 1] = self.pt2.vertices[7, 1]
        #         self.mesh.ad_points.pt[7, 1] = self.pt2.vertices[7, 1]
        #     if self.is_single == 2:
        #         self.mesh.ad_points.pt[7, 1] = self.pt1.vertices[8, 1]
        #         self.mesh.ad_points.pt[8, 1] = self.pt1.vertices[8, 1]

        # min_y = min(self.mesh_beiya.vertices[:, 1])
        # if (self.mesh.ad_points.pt[9][1] - min_y) < 0:
        #     self.mesh.ad_points.pt[9] -= self.axis_y / 2
        # else:
        #     self.mesh.ad_points.pt[9] -= (
        #         self.axis_y
        #         / np.linalg.norm(self.axis_y)
        #         * (self.mesh.ad_points.pt[9][1] - min_y + 0.5)
        #     )

        # points.extend(self.mesh.ad_points.pt)
        # points_id.extend(self.mesh.ad_points.idx)
        # points_id.extend(ray_id)
        # points.extend(p_z)
        # self.mesh = tps(self.mesh, points_id, points)

    def slice_mesh(self):
        self.axis = np.array(
            [
                self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1],
                self.mesh.cross_points.pt[2] - self.mesh.cross_points.pt[3],
            ]
        )

        self.axis = rotate_vector_to_90_degrees(self.axis)
        self.axis *= -1
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.print_matrix = rotate_from_axis(self.axis)

        self.neck_points, _ = find_boundaries(self.mesh_beiya)

        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.array(self.neck_points))
        )
        pcd.normals = o3d.utility.Vector3dVector(
            np.array([[0, 1, 0]]).repeat(len(self.neck_points), 0)
        )
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=3, linear_fit=False
        )[0]
        poisson_mesh = trimesh.Trimesh(
            np.asarray(poisson_mesh.vertices), np.asarray(poisson_mesh.triangles)
        )

        mesh_o3d = self.mesh.as_open3d
        distances_source, _ = compute_signed_distance(poisson_mesh, self.mesh.vertices)
        mesh_o3d.remove_vertices_by_index(np.where(distances_source > 0)[0])
        mesh_out = trimesh.Trimesh(mesh_o3d.vertices, mesh_o3d.triangles)
        interest_verts = self.mesh.nearest.vertex(mesh_out.vertices)[1]
        mesh = pypruners.pruner(
            self.mesh.vertices, self.mesh.faces, interest_verts, 1.0, -2.5, 1.5
        )
        mesh = get_biggest_mesh(mesh)
        self.mesh.update_mesh(mesh.vertices, mesh.faces)

        neck_p, self.neck_p_id = find_boundaries(self.mesh)

    def get_oppo_pre_dis(self):
        self.axis_y = np.array([0, 1, 0])
        points_dis_oppo = []
        points_dis_pre = []
        mesh_oppo = self.mesh_oppo.simplify_quadric_decimation(25000)

        neighbors_add = get_neighbors(self.mesh.ad_points.idx, 3, self.mesh)
        new_faces = np.array([x for x in mesh_oppo.faces if len(set(x)) == 3])
        mesh_oppo = trimesh.Trimesh(mesh_oppo.vertices, new_faces)
        for i in range(len(self.mesh.ad_points.idx)):
            ray_origins = copy.deepcopy(self.mesh.vertices[neighbors_add[i]])
            ray_origins += self.axis_y / np.linalg.norm(self.axis_y) * 10
            ray_directions = np.array([-self.axis_y]).repeat(
                len(neighbors_add[i]), axis=0
            )
            ray_out_oppo = trimesh.ray.ray_triangle.ray_triangle_id(
                mesh_oppo.triangles, ray_origins, ray_directions
            )
            if len(ray_out_oppo[0]):
                dis_oppo = np.max(
                    (
                        self.mesh.vertices[neighbors_add[i]][ray_out_oppo[1]]
                        - ray_out_oppo[2]
                    )[:, 1]
                )
                points_dis_oppo.append(dis_oppo)
            else:
                points_dis_oppo.append(-1000)

            ray_out_pre = trimesh.ray.ray_triangle.ray_triangle_id(
                self.mesh_beiya.triangles, ray_origins, ray_directions
            )
            if len(ray_out_pre[0]):
                dis_pre = np.min(
                    (
                        self.mesh.vertices[neighbors_add[i]][ray_out_pre[1]]
                        - ray_out_pre[2]
                    )[:, 1]
                )
                points_dis_pre.append(dis_pre)
            else:
                points_dis_pre.append(1000)
        return np.array(points_dis_oppo), np.array(points_dis_pre)

    def trans_oppo_new(self):
        self.mesh_copy = self.mesh.copy()
        points_dis_oppo, points_dis_pre = self.get_oppo_pre_dis()
        valid_indices = np.delete(np.arange(len(points_dis_pre)), (7, 8, 9))
        points_new = np.delete(self.mesh.ad_points.pt, (7, 8, 9), axis=0)
        if min(points_dis_pre[valid_indices]) < 0 and self.miss_id[1] == "7":
            if self.is_single == 1:
                points_8 = self.mesh.ad_points.pt[8]
                min_p_id = np.argmin(points_dis_pre[valid_indices])
                angle_oppo = angle_between_vectors(
                    points_new[min_p_id] - points_8,
                    points_new[min_p_id]
                    + (-points_dis_pre[valid_indices[min_p_id]])
                    * self.axis_y
                    / np.linalg.norm(self.axis_y)
                    - points_8,
                    2,
                )
                quaternion = trimesh.transformations.quaternion_about_axis(
                    angle_oppo, [0, 0, 1]
                )
                matrix = trimesh.transformations.quaternion_matrix(quaternion)
                self.mesh.apply_transform(matrix)
                self.mesh.apply_translation(
                    points_8 - self.mesh.vertices[self.mesh.ad_points.idx[8]]
                )
                self.mesh.update_mesh()
                points_dis_oppo, points_dis_pre = self.get_oppo_pre_dis()
            elif self.is_single == 2:
                points_7 = self.mesh.ad_points.pt[7]
                min_p_id = np.argmin(points_dis_pre[valid_indices])
                angle_oppo = angle_between_vectors(
                    points_new[min_p_id] - points_7,
                    points_new[min_p_id]
                    + (-points_dis_pre[valid_indices[min_p_id]])
                    * self.axis_y
                    / np.linalg.norm(self.axis_y)
                    - points_7,
                    2,
                )
                quaternion = trimesh.transformations.quaternion_about_axis(
                    -angle_oppo, [0, 0, 1]
                )
                matrix = trimesh.transformations.quaternion_matrix(quaternion)
                self.mesh.apply_transform(matrix)
                self.mesh.apply_translation(
                    points_7 - self.mesh.vertices[self.mesh.ad_points.idx[7]]
                )
                self.mesh.update_mesh()
                points_dis_oppo, points_dis_pre = self.get_oppo_pre_dis()
        if points_dis_oppo[8] > 1 and self.miss_id[1] == "7":
            points_7 = self.mesh.ad_points.pt[7]
            angle_oppo = angle_between_vectors(
                self.mesh.ad_points.pt[8] - points_7,
                self.mesh.ad_points.pt[8] - [0, points_dis_oppo[8], 0] - points_7,
                2,
            )
            quaternion = trimesh.transformations.quaternion_about_axis(
                angle_oppo, [0, 0, 1]
            )
            matrix = trimesh.transformations.quaternion_matrix(quaternion)
            self.mesh.apply_transform(matrix)
            self.mesh.apply_translation(
                points_7 - self.mesh.vertices[self.mesh.ad_points.idx[7]]
            )
            self.mesh.update_mesh()
            points_dis_oppo, points_dis_pre = self.get_oppo_pre_dis()
        points_new = np.delete(self.mesh.ad_points.pt, (7, 8, 9), axis=0)
        points_7 = self.mesh.ad_points.pt[7]
        points_8 = self.mesh.ad_points.pt[8]
        y_x = (points_7[1] - points_8[1]) / (points_7[0] - points_8[0])
        upper_y = (points_new[:, 0] - points_8[0]) * y_x + points_8[1]
        dis_max_y = points_new[:, 1] - upper_y
        upper_std = np.where(self.mesh.ad_points.pt[valid_indices, 1] > upper_y)[0]
        lower_std = np.where(self.mesh.ad_points.pt[valid_indices, 1] < upper_y)[0]
        if len(upper_std):
            if max(points_dis_oppo[valid_indices[upper_std]]) > 0:
                oppo_points_id = points_dis_oppo[valid_indices[upper_std]] > 0
                new_oppo_id = []
                if self.miss_id[-1] in ["6", "7"]:
                    for i in valid_indices[upper_std][oppo_points_id]:
                        for j in self.points_oppo_id:
                            if i in j:
                                for k in j:
                                    new_oppo_id.extend(
                                        np.where(valid_indices[upper_std] == k)[0]
                                    )

                oppo_points_id[new_oppo_id] = True
                oppo_std = max(
                    points_dis_oppo[valid_indices[upper_std]][oppo_points_id]
                    / dis_max_y[upper_std][oppo_points_id]
                )
                oppo_std = min(oppo_std, 0.8)

                self.mesh.ad_points.pt[valid_indices[upper_std][oppo_points_id], 1] -= (
                    self.mesh.ad_points.pt[valid_indices[upper_std][oppo_points_id], 1]
                    - upper_y[upper_std][oppo_points_id]
                ) * oppo_std
                if self.miss_id[-1] in ["4", "5"]:
                    self.mesh.ad_points.pt[
                        valid_indices[upper_std][np.logical_not(oppo_points_id)], 1
                    ] -= (
                        self.mesh.ad_points.pt[
                            valid_indices[upper_std][np.logical_not(oppo_points_id)], 1
                        ]
                        - upper_y[upper_std][np.logical_not(oppo_points_id)]
                    ) * oppo_std

        if len(lower_std):
            if min(points_dis_pre[valid_indices[lower_std]]) < 0.5:
                pre_points_id = points_dis_pre[valid_indices[lower_std]] < 0.5
                pre_std = max(
                    (points_dis_pre[valid_indices[lower_std]][pre_points_id] - 0.5)
                    / dis_max_y[lower_std][pre_points_id]
                )

                self.mesh.ad_points.pt[valid_indices[lower_std], 1] -= (
                    self.mesh.ad_points.pt[valid_indices[lower_std], 1]
                    - upper_y[lower_std]
                ) * min(pre_std, 0.9)
        points_id = []
        points = []
        points.extend(self.mesh.ad_points.pt)
        points_id.extend(self.mesh.ad_points.idx)
        points.extend(self.mesh.cross_points.pt[-2:])
        points_id.extend(self.mesh.cross_points.idx[-2:])
        self.mesh = tps(self.mesh, points_id, points, 0)

    def trans_oppo_from_area(self):
        self.mesh_copy = self.mesh.copy()

        self.axis_direct = np.array(
            [self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1]]
        )

        points_dis_oppo, points_dis_pre = self.get_oppo_pre_dis()
        valid_indices = np.delete(np.arange(len(points_dis_pre)), (7, 8, 9))
        points_new = np.delete(self.mesh.ad_points.pt, (7, 8, 9), axis=0)
        if min(points_dis_pre[valid_indices]) < 0 and self.miss_id[1] == "7":
            if self.is_single == 1:
                points_8 = self.mesh.ad_points.pt[8]
                min_p_id = np.argmin(points_dis_pre[valid_indices])
                angle_oppo = angle_between_vectors(
                    points_new[min_p_id] - points_8,
                    points_new[min_p_id]
                    + (-points_dis_pre[valid_indices[min_p_id]] + 0.3)
                    * 2
                    * self.axis_y
                    / np.linalg.norm(self.axis_y)
                    - points_8,
                    2,
                )
                quaternion = trimesh.transformations.quaternion_about_axis(
                    angle_oppo, [0, 0, 1]
                )
                matrix = trimesh.transformations.quaternion_matrix(quaternion)
                self.mesh.apply_transform(matrix)
                self.mesh.apply_translation(
                    points_8 - self.mesh.vertices[self.mesh.ad_points.idx[8]]
                )
                self.mesh.update_mesh()
            elif self.is_single == 2:
                points_7 = self.mesh.ad_points.pt[7]
                min_p_id = np.argmin(points_dis_pre[valid_indices])
                angle_oppo = angle_between_vectors(
                    points_new[min_p_id] - points_7,
                    points_new[min_p_id]
                    + (-points_dis_pre[valid_indices[min_p_id]] + 0.3)
                    * 2
                    * self.axis_y
                    / np.linalg.norm(self.axis_y)
                    - points_7,
                    2,
                )
                quaternion = trimesh.transformations.quaternion_about_axis(
                    -angle_oppo, [0, 0, 1]
                )
                matrix = trimesh.transformations.quaternion_matrix(quaternion)
                self.mesh.apply_transform(matrix)
                self.mesh.apply_translation(
                    points_7 - self.mesh.vertices[self.mesh.ad_points.idx[7]]
                )
                self.mesh.update_mesh()
        if points_dis_oppo[8] > 1 and self.miss_id[1] == "7":
            points_7 = self.mesh.ad_points.pt[7]
            angle_oppo = angle_between_vectors(
                self.mesh.ad_points.pt[8] - points_7,
                self.mesh.ad_points.pt[8] - [0, points_dis_oppo[8], 0] - points_7,
                2,
            )
            quaternion = trimesh.transformations.quaternion_about_axis(
                angle_oppo, [0, 0, 1]
            )
            matrix = trimesh.transformations.quaternion_matrix(quaternion)
            self.mesh.apply_transform(matrix)
            self.mesh.apply_translation(
                points_7 - self.mesh.vertices[self.mesh.ad_points.idx[7]]
            )
            self.mesh.update_mesh()

        self.axis_direct = np.array(
            [self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1]]
        )
        dis_oppo, closest_points = compute_signed_distance(
            self.mesh_oppo,
            self.mesh.vertices,
        )

        ng_points = np.where(dis_oppo > 0)[0]
        dis_y = closest_points[ng_points, 1] - self.mesh.vertices[ng_points, 1]
        ng_points = ng_points[np.where(dis_y < 0)[0]]
        ng_points = self.trans_oppo_step1(ng_points)
        if self.miss_id[-1] in ["6", "7"]:
            ng_points = self.trans_oppo_step2(ng_points)
            ng_points = self.trans_oppo_step3(ng_points)
        if self.miss_id[-1] in ["4", "5"]:
            ng_points = self.trans_oppo_step4(ng_points)

    def trans_oppo_step1(self, ng_points):
        if np.intersect1d(ng_points, self.mesh.oc_points.idx).size:
            loop_end = 1
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.o_points.idx,
                    self.mesh.o_points.pt,
                    condition_points=self.mesh.oc_points.idx,
                )
        return ng_points

    def trans_oppo_step2(self, ng_points):
        dis_o = trimesh.points.point_plane_distance(
            self.mesh.oc_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.gl_points.idx).size:
            dis_fc = trimesh.points.point_plane_distance(
                self.mesh.gl_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
            )
            max_iterations = (max(dis_fc[0]) - np.mean(dis_o[0])) // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                self.mesh.fc_points.idx,
                self.mesh.fc_points.pt,
                max_iterations,
                self.mesh.gl_points.idx,
            )
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.o_points.idx,
                    self.mesh.o_points.pt,
                    condition_points=self.mesh.gl_points.idx,
                )
        dis_o = trimesh.points.point_plane_distance(
            self.mesh.oc_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.gb_points.idx).size:
            dis_nfc = trimesh.points.point_plane_distance(
                self.mesh.gb_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
            )
            max_iterations = (max(dis_nfc[0]) - np.mean(dis_o[0])) // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                self.mesh.nfc_points.idx,
                self.mesh.nfc_points.pt,
                max_iterations,
                self.mesh.gb_points.idx,
            )
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.o_points.idx,
                    self.mesh.o_points.pt,
                    condition_points=self.mesh.gb_points.idx,
                )
        return ng_points

    def trans_oppo_step3(self, ng_points):
        dis_fc = trimesh.points.point_plane_distance(
            self.mesh.gl_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.fcmd_points.idx).size:
            dis_fcmd = trimesh.points.point_plane_distance(
                self.mesh.fcmd_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
            )
            max_iterations = (max(dis_fcmd[0]) - max(dis_fc[0])) / 2 // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                self.mesh.fcmd_points.idx,
                self.mesh.fcmd_points.pt,
                max_iterations,
                self.mesh.fcmd_points.idx,
                distance=0.05,
            )
            if loop_end:
                dis_o = trimesh.points.point_plane_distance(
                    self.mesh.oc_points.pt,
                    self.axis_direct,
                    self.mesh.cross_points.pt[1],
                )
                dis_fc = trimesh.points.point_plane_distance(
                    self.mesh.gl_points.pt,
                    self.axis_direct,
                    self.mesh.cross_points.pt[1],
                )
                max_iterations = (max(dis_fc[0]) - max(dis_o[0])) // 0.1
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.fc_points.idx,
                    self.mesh.fc_points.pt,
                    max_iterations,
                    self.mesh.fcmd_points.idx,
                    distance=0.05,
                )
                while loop_end:
                    ng_points, loop_end = self.trans_loop(
                        ng_points,
                        self.mesh.o_points.idx,
                        self.mesh.o_points.pt,
                        condition_points=self.mesh.fcmd_points.idx,
                        distance=0.05,
                    )
        dis_nfc = trimesh.points.point_plane_distance(
            self.mesh.gb_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.nfcmd_points.idx).size:
            dis_nfcmd = trimesh.points.point_plane_distance(
                self.mesh.nfcmd_points.pt,
                self.axis_direct,
                self.mesh.cross_points.pt[1],
            )
            max_iterations = (max(dis_nfcmd[0]) - max(dis_nfc[0])) / 2 // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                self.mesh.nfcmd_points.idx,
                self.mesh.nfcmd_points.pt,
                max_iterations,
                self.mesh.nfcmd_points.idx,
                distance=0.05,
            )
            if loop_end:
                dis_o = trimesh.points.point_plane_distance(
                    self.mesh.oc_points.pt,
                    self.axis_direct,
                    self.mesh.cross_points.pt[1],
                )
                dis_nfc = trimesh.points.point_plane_distance(
                    self.mesh.gb_points.pt,
                    self.axis_direct,
                    self.mesh.cross_points.pt[1],
                )
                max_iterations = (max(dis_nfc[0]) - max(dis_o[0])) // 0.1
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.nfc_points.idx,
                    self.mesh.nfc_points.pt,
                    max_iterations,
                    self.mesh.nfcmd_points.idx,
                    distance=0.05,
                )
                while loop_end:
                    ng_points, loop_end = self.trans_loop(
                        ng_points,
                        self.mesh.o_points.idx,
                        self.mesh.o_points.pt,
                        condition_points=self.mesh.nfcmd_points.idx,
                        distance=0.05,
                    )

        return ng_points

    def trans_oppo_step4(self, ng_points):
        dis_oc = trimesh.points.point_plane_distance(
            self.mesh.oc_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        dis_fc = trimesh.points.point_plane_distance(
            self.mesh.fc_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.fc_points.idx).size:
            max_iterations = (max(dis_fc[0]) - max(dis_oc[0])) / 2 // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                np.hstack((self.mesh.fc_points.idx, self.mesh.nfc_points.idx)),
                np.vstack((self.mesh.fc_points.pt, self.mesh.nfc_points.pt)),
                max_iterations,
                self.mesh.fc_points.idx,
                distance=0.05,
            )
        dis_nfc = trimesh.points.point_plane_distance(
            self.mesh.nfc_points.pt, self.axis_direct, self.mesh.cross_points.pt[1]
        )
        if np.intersect1d(ng_points, self.mesh.nfc_points.idx).size:
            max_iterations = (max(dis_nfc[0]) - max(dis_oc[0])) / 2 // 0.1
            ng_points, loop_end = self.trans_loop(
                ng_points,
                np.hstack((self.mesh.fc_points.idx, self.mesh.nfc_points.idx)),
                np.vstack((self.mesh.fc_points.pt, self.mesh.nfc_points.pt)),
                max_iterations,
                self.mesh.nfc_points.idx,
                distance=0.05,
            )
        return ng_points

    def trans_thickness_from_area(self):
        self.mesh_copy = self.mesh.copy()
        dis_pre, _ = compute_signed_distance(
            self.mesh_beiya,
            self.mesh.vertices,
        )
        ng_points = np.where(dis_pre > -0.3)[0]
        ng_points = np.intersect1d(ng_points, self.mesh.occl_points.idx)
        ng_points = self.trans_thickness_step1(ng_points)
        if self.miss_id[-1] in ["6", "7"]:
            ng_points = self.trans_thickness_step2(ng_points)

    def trans_thickness_step1(self, ng_points):
        if np.intersect1d(ng_points, self.mesh.oc_points.idx).size:
            loop_end = 1
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.o_points.idx,
                    self.mesh.o_points.pt,
                    condition_points=self.mesh.oc_points.idx,
                    distance=-0.3,
                    mode=1,
                )
            if len(ng_points):
                self.trans_thickness_final(self.mesh.oc_points.idx, ng_points)

        return ng_points

    def trans_thickness_step2(self, ng_points):
        if np.intersect1d(ng_points, self.mesh.gl_points.idx).size:
            loop_end = 1
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.fc_points.idx,
                    self.mesh.fc_points.pt,
                    condition_points=self.mesh.gl_points.idx,
                    distance=-0.3,
                    mode=1,
                )
            if len(ng_points):
                self.trans_thickness_final(
                    self.mesh.gl_points.idx,
                    ng_points,
                )
        if np.intersect1d(ng_points, self.mesh.gb_points.idx).size:
            loop_end = 1
            while loop_end:
                ng_points, loop_end = self.trans_loop(
                    ng_points,
                    self.mesh.nfc_points.idx,
                    self.mesh.nfc_points.pt,
                    condition_points=self.mesh.gb_points.idx,
                    distance=0,
                    mode=1,
                )
            if len(ng_points):
                self.trans_thickness_final(
                    self.mesh.gb_points.idx,
                    ng_points,
                )

        return ng_points

    def trans_thickness_final(self, p_idx, points_extend_idx):
        # p_idx = get_distance(self.mesh, p_idx.tolist(), 0.5)
        points_extend_idx = get_distance(self.mesh, points_extend_idx.tolist(), 1)
        points_extend_idx.extend(get_distance(self.mesh, p_idx.tolist(), 0.5))
        dis, _ = compute_signed_distance(
            self.mesh_beiya,
            self.mesh.vertices[p_idx],
        )
        dis = max(dis + 0.6)
        dis = np.array([0, dis, 0]).reshape(-1, 3).repeat(len(p_idx), axis=0)
        points_1 = self.mesh.vertices[p_idx]
        points_1 += dis
        points_id_1 = p_idx
        points_id_2 = np.setdiff1d(
            np.array(range(len(self.mesh.vertices))), np.array(points_extend_idx)
        )
        points_id_2 = random.sample(points_id_2.tolist(), len(points_id_2) // 10)
        points_2 = self.mesh.vertices[points_id_2]
        points = np.vstack([points_1, points_2])
        points_id = np.hstack([points_id_1, points_id_2])
        self.mesh = tps(self.mesh, points_id, points)

    def trans_loop(
        self,
        ng_points,
        points_id,
        points,
        max_iterations=10,
        condition_points=None,
        distance=0,
        mode=0,
    ):
        """
        迭代处理网格点的变换过程
        
        参数:
            ng_points: 需要处理的点索引数组
            points_id: 网格点的索引数组
            points: 网格点的坐标数组
            max_iterations: 最大迭代次数，默认为10
            condition_points: 条件点数组，默认为None（使用points）
            distance: 距离阈值，默认为0
            mode: 处理模式，0表示对侧处理，1表示背侧处理
            
        返回:
            ng_points: 处理后的点索引数组
            status: 状态码，1表示达到最大迭代次数，0表示正常完成
        """
        # 如果没有指定条件点，使用输入的点作为条件点
        if condition_points is None:
            condition_points = points
            
        iterations = 0
        # 将网格转换为Open3D格式
        mesh_o3d = self.mesh.as_open3d
        # 选择指定索引的点
        mesh_area = mesh_o3d.select_by_index(points_id)
        # 找到边界点和边界索引
        edge, edge_id = find_boundaries(mesh_area)
        # 创建点云对象
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(edge)))
        # 设置点云法向量
        pcd.normals = o3d.utility.Vector3dVector(self.axis_direct.repeat(len(edge), 0))
        # 使用泊松重建创建网格
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=4, width=0, scale=1.6, linear_fit=False
        )[0]
        # 将Open3D网格转换为trimesh格式
        poisson_mesh = trimesh.Trimesh(
            np.asarray(poisson_mesh.vertices), np.asarray(poisson_mesh.triangles)
        )
        
        # 迭代处理，直到满足条件或达到最大迭代次数
        while (
            np.intersect1d(ng_points, condition_points).size
            and iterations < max_iterations
        ):
            # 处理当前迭代
            self.trans_loop_process(points_id, points, poisson_mesh, mode)
            # 更新点坐标
            points = self.mesh.vertices[points_id]
            
            # 根据模式选择计算有符号距离的网格
            if mode:
                dis, _ = compute_signed_distance(
                    self.mesh_beiya,
                    self.mesh.vertices[ng_points],
                )
            else:
                dis, _ = compute_signed_distance(
                    self.mesh_oppo,
                    self.mesh.vertices[ng_points],
                )
                
            # 更新需要处理的点
            ng_points = ng_points[np.where(dis > distance)[0]]
            iterations += 1
            
        # 返回处理结果和状态
        if iterations >= max_iterations:
            return ng_points, 1  # 达到最大迭代次数
        else:
            return ng_points, 0  # 正常完成

    def trans_loop_process(self, points_id, points, poisson_mesh, mode=0):
        """
        处理单次迭代中的点变换过程
        
        参数:
            points_id: 网格点的索引数组
            points: 网格点的坐标数组
            poisson_mesh: 泊松重建后的网格
            mode: 处理模式，0表示咬合调整，1表示厚度调整
        """
        # 计算射线起点（沿法向量方向偏移）
        ray_origins = np.array(points) + self.axis_direct
        # 设置射线方向（与法向量相反）
        ray_directions = -self.axis_direct.repeat(len(points), 0)
        
        # 创建射线追踪场景
        scene = o3d.t.geometry.RaycastingScene()
        # 添加泊松重建的网格到场景中
        scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(poisson_mesh.as_open3d)
        )
        
        # 准备射线数据
        rays = np.hstack((ray_origins, ray_directions))
        # 执行射线追踪
        distance = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.float32))
        # 计算实际距离（减去初始偏移）
        distance = distance["t_hit"].numpy() - 1
        # 处理无效距离
        distance[distance < 0] = 0
        # 处理无穷大距离
        if np.isinf(max(distance)):
            distance[np.where(distance == max(distance))[0]] = 0
            
        # 计算位移向量
        distance = ray_directions * distance.repeat(3).reshape(-1, 3)
        # 归一化位移向量
        if np.max(np.linalg.norm(distance, axis=1)) == 0:
            distance = ray_directions / np.max(np.linalg.norm(ray_directions, axis=1))
        else:
            distance /= np.max(np.linalg.norm(distance, axis=1))
            
        # 调整射线起点
        ray_origins -= self.axis_direct
        
        # 根据模式计算新的点位置
        if mode:
            # 厚度调整模式：向内移动
            points = ray_origins - distance * 0.1
        else:
            # 咬合调整模式：向外移动
            points = ray_origins + distance * 0.1
            
        # 更新网格顶点
        m_points = np.array(self.mesh.vertices)
        m_points[points_id] = points
        self.mesh.update_mesh(m_points)

    def rebuild_y(self):
        valid_indices = np.delete(np.arange(len(self.mesh.ad_points.pt)), 9)
        neighbors_add = get_distance(
            self.mesh, np.array(self.mesh.ad_points.idx)[valid_indices].tolist(), 2
        )
        ray_origins = copy.deepcopy(self.mesh.vertices[neighbors_add])
        ray_directions = np.array([self.axis_y]).repeat(len(neighbors_add), axis=0)
        mesh_beiya = self.mesh_beiya.simplify_quadric_decimation(25000)
        new_faces = np.array([x for x in mesh_beiya.faces if len(set(x)) == 3])
        mesh_beiya = trimesh.Trimesh(mesh_beiya.vertices, new_faces)
        ray_out_beiya = trimesh.ray.ray_triangle.ray_triangle_id(
            mesh_beiya.triangles, ray_origins, ray_directions
        )

        if len(ray_out_beiya[0]):
            z = self.mesh.vertices[neighbors_add][ray_out_beiya[1]][:, 2]
            k_z = len(np.where(z > 0)[0]) / len(z)
            if k_z > 0.9 or k_z < 0.1:
                angles = [
                    angle_between_vectors(
                        self.mesh.vertices[neighbors_add][ray_out_beiya[1]][i],
                        ray_out_beiya[2][i] + [0, 0.6, 0],
                        0,
                    )
                    for i in range(len(ray_out_beiya[0]))
                ]
                if k_z < 0.1:
                    angle_beiya = max(angles)
                else:
                    angle_beiya = -max(angles)
                quaternion = trimesh.transformations.quaternion_about_axis(
                    angle_beiya, [1, 0, 0]
                )
                matrix = trimesh.transformations.quaternion_matrix(quaternion)
                self.mesh.apply_transform(matrix)
                self.mesh.update_mesh()
                self.trans_oppo_new()
            else:
                dis_oppo = (
                    np.max(
                        (
                            ray_out_beiya[2]
                            - self.mesh.vertices[neighbors_add][ray_out_beiya[1]]
                        )[:, 1]
                    )
                    + self.minimal_thickness
                )
                self.mesh.apply_translation([0, dis_oppo, 0])
                self.mesh.update_mesh()

    def trans_linya(self):
        # 如果输入的是没有虚拟牙根的牙冠模型, 邻接区域需要避开牙冠边缘
        if self.mesh.outline().discrete:
            neck_p = self.mesh.outline().discrete[0]
            _, neck_p_id = find_new_points(self.mesh, neck_p, 1)
            self.mesh.add_keypoint("neck_p", neck_p_id)
            neck_p = get_distance(self.mesh, neck_p_id, 1)
            ad_p_1 = np.array([x for x in self.mesh.adj_points1.idx if x not in neck_p])
            ad_p_2 = np.array([x for x in self.mesh.adj_points2.idx if x not in neck_p])
        else:
            ad_p_1 = self.mesh.adj_points1.idx
            ad_p_2 = self.mesh.adj_points2.idx

        # 分别计算邻牙与对应邻接区域的距离
        _, closest_p_1 = compute_signed_distance(self.mesh1, self.mesh.vertices[ad_p_1])
        _, closest_p_2 = compute_signed_distance(self.mesh2, self.mesh.vertices[ad_p_2])
        # 找到距离最近的点的索引
        dis_max_1 = np.argmin((closest_p_1 - self.mesh.vertices[ad_p_1])[:, 0])
        dis_max_2 = np.argmax((closest_p_2 - self.mesh.vertices[ad_p_2])[:, 0])
        # 根据索引得到最近点连线的向量
        ad_arr_1 = closest_p_1[dis_max_1] - self.mesh.vertices[ad_p_1[dis_max_1]]
        ad_arr_2 = closest_p_2[dis_max_2] - self.mesh.vertices[ad_p_2[dis_max_2]]
        # 记录牙冠上最近点的索引id
        self.ad_id_1 = ad_p_1[dis_max_1]
        self.ad_id_2 = ad_p_2[dis_max_2]
        # 根据索引id找邻域作为变换区域
        ad_ids_1 = get_neighbors([self.ad_id_1], 3, self.mesh)[0]
        ad_ids_2 = get_neighbors([self.ad_id_2], 3, self.mesh)[0]
        # 取邻域与邻接区域的交集, 保证变换区域在邻接区域中
        ad_ids_1 = np.intersect1d(ad_ids_1, ad_p_1)
        ad_ids_2 = np.intersect1d(ad_ids_2, ad_p_2)
        # 根据最近点连线的向量方向计算每个点需要移动的距离
        ray_origins = copy.deepcopy(self.mesh.vertices[ad_ids_1] - ad_arr_1)
        ray_directions = np.array([ad_arr_1]).repeat(len(ad_ids_1), axis=0)
        ray_ad_1 = trimesh.ray.ray_triangle.ray_triangle_id(
            self.mesh1.triangles, ray_origins, ray_directions
        )

        ray_origins = copy.deepcopy(self.mesh.vertices[ad_ids_2] - ad_arr_2)
        ray_directions = np.array([ad_arr_2]).repeat(len(ad_ids_2), axis=0)
        ray_ad_2 = trimesh.ray.ray_triangle.ray_triangle_id(
            self.mesh2.triangles, ray_origins, ray_directions
        )
        # 设置阈值, 保证找到的点是正确的
        ray_id_1 = np.where(
            np.linalg.norm(ray_ad_1[2] - closest_p_1[dis_max_1], axis=1) < 1
        )
        ray_id_2 = np.where(
            np.linalg.norm(ray_ad_2[2] - closest_p_2[dis_max_2], axis=1) < 1
        )
        # 根据交点信息得到目标点
        ad_points_1 = ray_ad_1[2][ray_id_1]
        ad_points_2 = ray_ad_2[2][ray_id_2]

        ad_p_1_ = ad_ids_1[ray_ad_1[1][ray_id_1]]
        ad_p_2_ = ad_ids_2[ray_ad_2[1][ray_id_2]]
        # 计算需要移动的距离
        dis_1 = (
            np.linalg.norm(ad_points_1 - self.mesh.vertices[ad_p_1_], axis=1)
            + self.ad_gap
        )
        dis_2 = (
            np.linalg.norm(ad_points_2 - self.mesh.vertices[ad_p_2_], axis=1)
            + self.ad_gap
        )
        # 计算最终的目标点位
        ad_points_1 = self.mesh.vertices[ad_p_1_] + ad_arr_1 / np.linalg.norm(
            ad_arr_1
        ) * dis_1.reshape((-1, 1))
        ad_points_2 = self.mesh.vertices[ad_p_2_] + ad_arr_2 / np.linalg.norm(
            ad_arr_2
        ) * dis_2.reshape((-1, 1))

        # ad_points_1 = (self.mesh.vertices[ad_p_1] + ad_arr_1).tolist()
        # ad_points_2 = (self.mesh.vertices[ad_p_2] + ad_arr_2).tolist()
        # 单邻牙情况下只对一侧邻牙做邻接调整
        if int(self.miss_id) in [17, 27, 37, 47]:
            ad_points = []
            ad_points.extend(ad_points_1)
            ad_points.extend(self.mesh.vertices[ad_p_2])
            ad_points_id = []
            ad_points_id.extend(ad_p_1_)
            ad_points_id.extend(ad_p_2)
            self.mesh.add_keypoint("linya_points", [self.ad_id_1])
        elif self.is_single == 1:
            ad_points = []
            ad_points.extend(self.mesh.vertices[ad_p_1])
            ad_points.extend(ad_points_2)
            ad_points_id = []
            ad_points_id.extend(ad_p_1)
            ad_points_id.extend(ad_p_2_)
            self.mesh.add_keypoint("linya_points", [self.ad_id_2])
        elif self.is_single == 2:
            ad_points = []
            ad_points.extend(ad_points_1)
            ad_points.extend(self.mesh.vertices[ad_p_2])
            ad_points_id = []
            ad_points_id.extend(ad_p_1_)
            ad_points_id.extend(ad_p_2)
            self.mesh.add_keypoint("linya_points", [self.ad_id_1])
        else:
            ad_points = []
            ad_points.extend(ad_points_1)
            ad_points.extend(ad_points_2)
            ad_points_id = []
            ad_points_id.extend(ad_p_1_)
            ad_points_id.extend(ad_p_2_)
            self.mesh.add_keypoint("linya_points", [self.ad_id_1, self.ad_id_2])

        points_id = []
        points = []
        points_id.extend(ad_points_id)
        points.extend(ad_points)
        points_id.extend(self.mesh.cross_points.idx[1:2])
        points.extend(self.mesh.cross_points.pt[1:2])
        self.mesh = tps(self.mesh, points_id, points)

    def trans_pre_oppo(self):
        for i in range(20):
            mesh_o3d = self.mesh.copy().as_open3d
            for _ in range(3):
                mesh_o3d.remove_vertices_by_index(
                    np.asarray(
                        mesh_o3d.get_non_manifold_edges(allow_boundary_edges=False)
                    ).flatten()
                )
            distances, _ = compute_signed_distance(
                self.mesh_oppo, np.asarray(mesh_o3d.vertices)
            )
            mesh_o3d_pre = copy.deepcopy(mesh_o3d)
            pre_points = []
            pre_points_id = []
            if len(np.where(distances > -self.occlusal_distance)[0]):
                mesh_o3d_pre.remove_vertices_by_index(
                    np.where(distances < -self.occlusal_distance)[0]
                )
                mesh_hole = trimesh.Trimesh(
                    mesh_o3d_pre.vertices, mesh_o3d_pre.triangles
                )
                mesh_holes = mesh_hole.split(only_watertight=False)
                if (
                    len(mesh_holes)
                    and len(mesh_hole.vertices) / len(self.mesh.vertices) < 0.5
                ):
                    for mh in mesh_holes:
                        mh_dis, mh_ans = compute_signed_distance(
                            self.mesh_oppo, np.asarray(mh.vertices)
                        )
                        mh_dis_pre, mh_ans_pre = compute_signed_distance(
                            self.mesh_beiya, np.asarray(mh.vertices)
                        )
                        # if max(mh_dis_pre) > -self.minimal_thickness:
                        #     continue
                        if min(mh_dis) > 1:
                            continue
                        if len(mh_dis) > 100:
                            distan = np.linalg.norm(mh.vertices, axis=1)
                            total_distan = np.sum(distan)
                            weights = distan / total_distan
                            sampled_indices = np.random.choice(
                                len(mh.vertices),
                                size=len(mh.vertices) // 100,
                                replace=False,
                                p=weights,
                            )
                            sampled_points = mh.vertices[sampled_indices]
                        else:
                            sampled_indices = [np.argmax(mh_dis)]
                            sampled_points = mh.vertices[sampled_indices]
                        # dis, ans = compute_signed_distance(
                        #     self.mesh_oppo, sampled_points
                        # )
                        dis, ans = mh_dis[sampled_indices], mh_ans[sampled_indices]
                        _dis_pre, _ans_pre = (
                            mh_dis_pre[sampled_indices],
                            mh_ans_pre[sampled_indices],
                        )
                        for idx in range(len(sampled_points)):
                            if dis[idx] < 0:
                                dis_a = -self.occlusal_distance - dis[idx]
                                # dis_a = min(dis_a, -dis_pre[idx] - self.minimal_thickness)
                            else:
                                dis_a = -(-self.occlusal_distance - dis[idx])
                                # dis_a = min(dis_a, -dis_pre[idx] - self.minimal_thickness)
                            pre_p = (
                                sampled_points[idx]
                                + np.array(
                                    (ans[idx] - sampled_points[idx])
                                    / np.linalg.norm(ans[idx] - sampled_points[idx])
                                    * dis_a
                                )[1]
                                * self.axis_y
                                / self.axis_y[1]
                            )
                            pre_points.append(pre_p)
                            unique_elements, counts = np.unique(
                                np.where(sampled_points[idx] == self.mesh.vertices)[0],
                                return_counts=True,
                            )
                            pre_points_id.append(unique_elements[np.argmax(counts)])
                    _, trans_points_id = find_new_points(
                        self.mesh, mesh_hole.vertices, 1
                    )
                    trans_points_id = get_distance(self.mesh, trans_points_id, 3)
                    safe_points_id = [
                        x
                        for x in range(len(self.mesh.vertices))
                        if x not in trans_points_id
                    ]
                    safe_points_id = random.sample(
                        safe_points_id, len(safe_points_id) // 10
                    )
                    pre_points.extend(self.mesh.cross_points.pt[-2:])
                    pre_points_id.extend(self.mesh.cross_points.idx[-2:])
                    pre_points.extend(self.mesh.vertices[safe_points_id])
                    pre_points_id.extend(safe_points_id)
                    self.mesh = tps(self.mesh, pre_points_id, pre_points)
                else:
                    break
            else:
                break

    def trans_pre_thickness(self):
        outer_verts = self.mesh.vertices
        outer_faces = self.mesh.faces
        inner_verts = self.mesh_beiya.vertices
        inner_faces = self.mesh_beiya.faces
        if "linya_points" in vars(self.mesh):
            if self.prox_or_occlu not in [0, 2]:
                linya_points_ply = trimesh.PointCloud(self.mesh.linya_points.pt)
                linya_points_ply.apply_transform(self.trans_matrix)
                self.linya_points = np.array(linya_points_ply.vertices)
            ad_points, ad_points_id = find_new_points(
                self.mesh, self.mesh.linya_points.pt, 1
            )
            ad_neighbor_id = get_distance(self.mesh, ad_points_id, 1)
            ad_points_id.extend(ad_neighbor_id)
            ad_points_id = list(set(ad_points_id))
            ad_points = self.mesh.vertices[ad_points_id]
        if self.add_point_id:
            pin_points = np.array([])
            pin_points = np.array(
                self.mesh.vertices[np.delete(self.add_point_id, 9, 0)]
            )
            if "linya_points" in vars(self.mesh):
                pin_points = np.concatenate([pin_points, ad_points], axis=0)
        else:
            if "linya_points" in vars(self.mesh):
                pin_points = ad_points
            else:
                pin_points = np.array([])
        import py_minimum_thickness

        self.thickness_face_id = py_minimum_thickness.findThinkness(
            outer_verts,
            outer_faces,
            inner_verts,
            inner_faces,
            self.minimal_thickness,
        )

        if len(self.thickness_face_id):
            if self.thick_flag:
                new_points = py_minimum_thickness.crownAdaptive(
                    self.minimal_thickness,
                    outer_verts,
                    outer_faces,
                    inner_verts,
                    inner_faces,
                    pin_points,
                )
                if self.handler_name == "post":
                    self.mesh.update_mesh(new_points)
                elif self.handler_name == "occlu":
                    self.mesh_adapt_thickness = TrackedTrimesh(new_points, outer_faces)
                    for v in vars(self.mesh):
                        if v not in vars(trimesh.Trimesh()) and v not in [
                            "keypoints",
                            "vert_num",
                        ]:
                            self.mesh_adapt_thickness.add_keypoint(
                                v, getattr(self.mesh, v).idx
                            )
                    changed_faces = find_changed_faces(
                        self.mesh, self.mesh_adapt_thickness
                    )

                    self.thickness_face_id = remove_boundary_faces(
                        self.mesh, changed_faces
                    )
            else:
                self.thickness_face_id = remove_boundary_faces(
                    self.mesh, self.thickness_face_id.reshape(-1)
                )

    def trans_neck(self):
        mesh = pylfda.Mesh()
        mesh.vertices, mesh.faces = self.mesh.vertices, self.mesh.faces
        if "linya_points" in vars(self.mesh):
            ad_points_id = self.mesh.linya_points.idx.tolist()
        else:
            self.mesh.ad_points.update(
                np.delete(self.mesh.ad_points.idx, (9), axis=0),
                np.delete(self.mesh.ad_points.pt, (9), axis=0),
            )
            ad_points_id = self.mesh.ad_points.idx.tolist()
        ad_neighbor_id = get_distance(self.mesh, ad_points_id, 0.5)
        ad_points_id.extend(ad_neighbor_id)
        ad_points_id = list(set(ad_points_id))
        ad_points = self.mesh.vertices[ad_points_id]
        neck_points = get_edges.prep_edge_smoothing.move_outward(
            self.neck_points, self.margin_width
        )
        neck_p, self.neck_p_id = find_boundaries(self.mesh)
        # neck_p = self.mesh.outline().discrete[0]
        # neck_p, self.neck_p_id = find_new_points(self.mesh, neck_p, 1)
        neck_points_id, neck_points = find_edge_st_mesh(
            neck_points, self.mesh.vertices[self.neck_p_id], self.mesh
        )  # 查找标准牙冠与备牙边缘对应点的索引
        neck_neighbor_id = get_distance(self.mesh, neck_points_id, 4)
        points_p_id = [
            x for x in range(len(self.mesh.vertices)) if x not in neck_neighbor_id
        ]

        points_p_id = random.sample(points_p_id, len(points_p_id) // 10)
        points_p = self.mesh.vertices[points_p_id]
        points = []
        points_id = []
        points.extend(neck_points)
        points_id.extend(neck_points_id)
        points.extend(points_p)
        points_id.extend(points_p_id)
        points.extend(ad_points)
        points_id.extend(ad_points_id)

        self.mesh = tps(self.mesh, points_id, points)
        

    def fill_gap(self):
        mesh_inner = self.mesh_beiya.copy()
        mesh_inner.invert()
        if self.handler_name == "occlu":
            if self.thick_flag:
                if len(self.thickness_face_id) and self.mesh_without_thickness:
                    # changed_faces = find_changed_faces(self.ori_mesh, self.mesh)
                    # self.thickness_face_id = remove_boundary_faces(self.mesh, changed_faces)
                    self.thickness_points_id = self.mesh.faces[
                        self.thickness_face_id
                    ].reshape(-1)
                    self.thickness_points = self.mesh.vertices[self.thickness_points_id]
                else:
                    self.without_thickness_add_points = self.mesh.ad_points.pt
                    self.without_thickness_cross_points = self.mesh.cross_points.pt
                    self.without_thickness_adj_points1 = self.mesh.adj_points1.pt
                    self.without_thickness_adj_points2 = self.mesh.adj_points2.pt

            self.ori_mesh = self.mesh.copy()
        mesh = pylfda.Mesh()
        mesh.vertices, mesh.faces = self.mesh.vertices, self.mesh.faces
        mex_edge_length = 0.15
        pylfda.subdivide_mesh(mesh, mex_edge_length)
        print(f'subdivide_mesh end. p_num:{len(mesh.vertices)}, f_num:{len(mesh.faces)}')
        # desired_count = len(mesh.vertices) // 2
        desired_count = 12500
        decimationType = pylfda.DecimationType.Vertex
        max_normal_deviation = 1
        fix_boundary = False
        out = pylfda.decimate_mesh(
            mesh, desired_count, decimationType, max_normal_deviation, fix_boundary
        )
        if out:
            self.mesh.update_mesh(mesh.vertices, mesh.faces)
        print(f'decimate_mesh end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}')
        mesh = self.mesh.subdivide_loop(1)
        self.mesh.update_mesh(mesh.vertices, mesh.faces)
        print(f'subdivide_loop end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}')
        mesh = self.mesh.as_open3d
        # mesh.merge_close_vertices(0.05)
        mesh.remove_non_manifold_edges()
        print(f'remove_non_manifold_edges end. p_num:{len(np.asarray(mesh.vertices))}, f_num:{len(np.asarray(mesh.triangles))}')
        while True:
            non_mani_p = mesh.get_non_manifold_vertices()
            if len(non_mani_p):
                mesh.remove_vertices_by_index(non_mani_p)
            else:
                break
        self.mesh.update_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        print(f'remove_vertices_by_index end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}')
        self.mesh_outside = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces)
        # 备牙修复
        # beiya_repair = pymeshrepairer.Mesh()
        # beiya_repair.vertices = mesh_inner.vertices
        # beiya_repair.faces = mesh_inner.faces
        # pymeshrepairer.repair(beiya_repair)
        mesh_inner = mesh_inner.as_open3d
        mesh_inner.remove_degenerate_triangles()
        mesh_inner.remove_non_manifold_edges()
        mesh_inner = trimesh.Trimesh(mesh_inner.vertices, mesh_inner.triangles)

        # 备牙补洞
        beiya_lfda = pylfda.Mesh()
        # beiya_lfda.vertices = beiya_repair.vertices
        # beiya_lfda.faces = beiya_repair.faces
        beiya_lfda.vertices = mesh_inner.vertices
        beiya_lfda.faces = mesh_inner.faces
        maximum_filling_hole_size = 2
        pylfda.fill_hole(beiya_lfda, maximum_filling_hole_size)
        mesh_inner = trimesh.Trimesh(beiya_lfda.vertices, beiya_lfda.faces)

        mesh_inner = mesh_inner.as_open3d
        # mesh_inner.merge_close_vertices(0.1)
        mesh_inner.compute_vertex_normals()
        mesh_inner.remove_duplicated_vertices()
        mesh_inner.remove_degenerate_triangles()
        mesh_inner.remove_unreferenced_vertices()
        mesh_inner = trimesh.Trimesh(mesh_inner.vertices, mesh_inner.triangles)
        # mesh_inner.merge_vertices(merge_norm=True)
        mesh_o3d = self.mesh.as_open3d
        mesh_o3d.compute_vertex_normals()
        print(f'compute_vertex_normals end. p_num:{len(np.asarray(mesh_o3d.vertices))}, f_num:{len(np.asarray(mesh_o3d.triangles))}')
        mesh_o3d.remove_duplicated_vertices()
        print(f'remove_duplicated_vertices end. p_num:{len(np.asarray(mesh_o3d.vertices))}, f_num:{len(np.asarray(mesh_o3d.triangles))}')
        mesh_o3d.remove_degenerate_triangles()
        print(f'remove_degenerate_triangles end. p_num:{len(np.asarray(mesh_o3d.vertices))}, f_num:{len(np.asarray(mesh_o3d.triangles))}')
        mesh_o3d.remove_unreferenced_vertices()
        print(f'remove_unreferenced_vertices end. p_num:{len(np.asarray(mesh_o3d.vertices))}, f_num:{len(np.asarray(mesh_o3d.triangles))}')
        self.mesh.update_mesh(
            np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles)
        )
        print(f'repair end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}')
        # mesh_concat = trimesh.util.concatenate([mesh_inner, self.mesh])
        mesh0 = pylfda.Mesh()
        mesh1 = pylfda.Mesh()
        mesh0.vertices, mesh0.faces = self.mesh.vertices, self.mesh.faces
        mesh1.vertices, mesh1.faces = mesh_inner.vertices, mesh_inner.faces

        mesh_out = pylfda.Mesh()
        max_stitching_distance = 1
        self.stitch_success = pylfda.stitch(
            mesh1, mesh0, mesh_out, max_stitching_distance
        )
        self.mesh.update_mesh(mesh_out.vertices, mesh_out.faces)
        # self.undercut_mesh = trimesh.boolean.difference([self.mesh, self.undercut_mesh])
        # self.mesh.export('m.stl')
        if self.handler_name != "stitch":
            if self.mesh_without_thickness:
                self.without_thickness_add_points, add_normal_id = find_new_points(
                    self.mesh_without_thickness, self.without_thickness_add_points, 1
                )
                mesh_o3d = self.mesh_without_thickness.as_open3d
                mesh_o3d.compute_vertex_normals()
                self.without_thickness_add_point_normal = np.asarray(
                    mesh_o3d.vertex_normals
                )[add_normal_id]
                self.without_thickness_add_point_normal = (
                    self.without_thickness_add_point_normal
                    / np.linalg.norm(
                        self.without_thickness_add_point_normal, axis=1, keepdims=True
                    )
                )
            mesh_o3d = self.mesh.as_open3d
            mesh_o3d.compute_vertex_normals()
            self.add_point_normal = np.asarray(mesh_o3d.vertex_normals)[
                self.mesh.ad_points.idx
            ]
            self.add_point_normal = self.add_point_normal / np.linalg.norm(
                self.add_point_normal, axis=1, keepdims=True
            )

    def trans_neck2(self):
        neck_points, _ = find_boundaries(self.mesh_beiya)
        # neck_points = self.mesh_beiya.vertices[self.mesh_beiya.outline().referenced_vertices]
        neck_points = get_edges.prep_edge_smoothing.move_outward(
            neck_points, self.margin_width
        )
        neck_p, _ = find_boundaries(self.mesh)
        # neck_p = self.mesh.vertices[self.mesh.outline().referenced_vertices]
        # neck_p, neck_p_id = find_new_points(self.mesh, neck_p, 1)
        neck_points_id, neck_points = find_edge_st_mesh(neck_points, neck_p, self.mesh)
        # 查找标准牙冠与备牙边缘对应点的索引

        neck_neighbor_id = get_distance(self.mesh, neck_points_id, 4)
        points_p_id = [
            x for x in range(len(self.mesh.vertices)) if x not in neck_neighbor_id
        ]
        points_p_id = random.sample(points_p_id, len(points_p_id) // 10)
        points_p = self.mesh.vertices[points_p_id]
        points = []
        points_id = []
        points.extend(neck_points)
        points_id.extend(neck_points_id)
        points.extend(points_p)
        points_id.extend(points_p_id)
        self.mesh = tps(self.mesh, points_id, points)

    def simplify(self, point_num=8000, update=False):
        # self.mesh.apply_transform(self.trans_matrix)
        point_num = min(point_num, len(self.mesh.vertices))
        if update:
            mesh = self.mesh.simplify_quadric_decimation(point_num)
            self.mesh.update_mesh(mesh.vertices, mesh.faces)
        else:
            self.mesh = self.mesh.simplify_quadric_decimation(point_num)
            
        mesh = pylfda.Mesh()
        mesh.vertices, mesh.faces = self.mesh.vertices, self.mesh.faces
        mex_edge_length = 0.4
        pylfda.subdivide_mesh(mesh, mex_edge_length)
        desired_count = len(mesh.vertices) // 2
        decimationType = pylfda.DecimationType.Vertex
        max_normal_deviation = 1
        fix_boundary = True
        out = pylfda.decimate_mesh(
            mesh, desired_count, decimationType, max_normal_deviation, fix_boundary
        )
        if out:
            if update:
                self.mesh.update_mesh(mesh.vertices, mesh.faces)
            else:
                self.mesh = TrackedTrimesh(mesh.vertices, mesh.faces)

    def revert_mesh(self):
        if (self.pt1 and self.pt2) or self.AB:
            self.mesh.apply_transform(np.linalg.pinv(self.pt_matrix))

        self.mesh.apply_transform(np.linalg.pinv(self.rotation_matrix))
        self.mesh.apply_transform(np.linalg.pinv(self.matrix))

        if self.transform:
            self.mesh.apply_transform(np.linalg.pinv(self.transform))

    def get_std_crown(self):
        self.read_mesh()
        print("read_mesh end")

        self.axis_y = self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1]

        self.get_matrix_from_mesh()
        self.get_matrix_from_pt()
        self.get_matrix_from_ai()
        print("transform end")

        self.axis_y = self.mesh.cross_points.pt[0] - self.mesh.cross_points.pt[1]

        self.get_adjacent_area()

        self.get_roi_from_upper_lower()
        print("get_roi_from_upper_lower end")

        # self.get_purue_beiya()
        # print("get_purue_beiya end")

        # self.trans_y()
        # print("trans_y end")
        self.trans_scale()
        print("trans_scale end")
        self.trans_p()
        print("trans_p end")

    def get_post_mesh(self):
        self.load_paras()
        print(f"load_paras end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.load_points()
        print(f"load_config end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        if self.pre_tag == "std":
            self.get_purue_beiya()
            print(f"get_purue_beiya end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
            # self.neck_points, _ = find_boundaries(self.mesh_beiya)
            # if self.adjust_crown:
            self.dilation()
            print(f"dilation end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.mesh_beiya = remove_degenerate_faces(self.mesh_beiya)
        print(f'adjust_crown is {self.adjust_crown}')
        if self.adjust_crown:
            print('start adj')
            self.trans_linya()
            if self.template_name == "st_tooth":
                self.trans_oppo_from_area()
                print(f"trans_oppo end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
                self.trans_thickness_from_area()
                print(f"trans_thickness end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
            else:
                self.trans_oppo_new()
                print(f"trans_oppo end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
                self.rebuild_y()
                print(f"rebuild_y end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
            self.trans_linya()
            print(f"trans_linya end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
            self.trans_pre_oppo()
            print(f"trans_pre_oppo end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.trans_pre_thickness()
        print(f"trans_pre_thickness end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.slice_mesh()
        print(f"slice_mesh end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.trans_neck()
        print(f"trans_neck end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        self.fill_gap()
        print(f"fill_gap end. p_num:{len(self.mesh.vertices)}, f_num:{len(self.mesh.faces)}")
        # self.simplify(point_num=50000, update=True)

    def get_adjust_mesh(self):
        # self.load_paras()
        print("load_paras end")
        if self.align_edges:
            self.trans_neck2()
            print("trans_neck2 end")
        self.fill_gap()
        print("fill_gap end")

    def get_occlu_mesh(self):
        self.load_paras()
        self.simplify()
        self.load_points()
        self.mesh.update_mesh()

        self.axis_y = np.array([0, 1, 0])
        print("load_paras end")
        print(f'prox_or_occlu is {self.prox_or_occlu}')
        if self.prox_or_occlu in [0, 2]:
            print('start adj')
            self.trans_linya()
        if self.prox_or_occlu in [1, 2]:
            print('start oppo')
            self.trans_pre_oppo()
        print("trans_pre_oppo end")
        self.trans_pre_thickness()
        print("trans_pre_thickness end")

        self.get_adjust_mesh()

        if not self.thick_flag:
            if len(self.thickness_face_id):
                original_to_subdivided = find_subdivided_faces(
                    self.ori_mesh.vertices,
                    self.ori_mesh.faces[self.thickness_face_id],
                    self.mesh.vertices,
                    self.mesh.faces,
                )
                self.thickness_points_id = np.unique(
                    np.array([z for x in original_to_subdivided for y in x for z in y])
                )
                self.thickness_points_id = self.thickness_points_id[
                    remove_intersection(
                        self.mesh.vertices[self.thickness_points_id],
                        self.mesh_beiya.vertices,
                    )
                ]
            return

        self.mesh_outside_without_thickness = self.mesh_outside.copy()
        self.mesh_without_thickness = trimesh.Trimesh(
            self.mesh.vertices, self.mesh.faces
        )

        if len(self.thickness_face_id):
            self.mesh = TrackedTrimesh(
                self.mesh_adapt_thickness.vertices, self.mesh_adapt_thickness.faces
            )
            for v in vars(self.mesh_adapt_thickness):
                if v not in vars(trimesh.Trimesh()) and v not in [
                    "keypoints",
                    "vert_num",
                ]:
                    self.mesh.add_keypoint(v, getattr(self.mesh_adapt_thickness, v).idx)
            self.get_adjust_mesh()
            _, self.thickness_points_id = find_new_points(
                self.ori_mesh, self.thickness_points, 0
            )
            self.thickness_points_id = np.array(self.thickness_points_id).reshape(
                (-1, 3)
            )
            original_to_subdivided = find_subdivided_faces(
                self.ori_mesh.vertices,
                self.thickness_points_id,
                self.mesh.vertices,
                self.mesh.faces,
            )
            self.thickness_points_id = np.unique(
                np.array([z for x in original_to_subdivided for y in x for z in y])
            )
            self.thickness_points_id = self.thickness_points_id[
                remove_intersection(
                    self.mesh.vertices[self.thickness_points_id],
                    self.mesh_beiya.vertices,
                )
            ]

    def stitch_edge(self):
        self.load_paras()
        print("load_paras end")
        self.simplify()

        if self.align_edges:
            self.get_purue_beiya()
            # self.neck_points, _ = find_boundaries(self.mesh_beiya)
            self.dilation()
            print("dilation end")
            self.trans_neck2()
            print("trans_neck2 end")
        self.fill_gap()
        print("fill_gap end")

    def undercpt_filling(self):
        self.load_paras()
        print("load_paras end")
        if len(self.mesh_beiya.faces) > 8000:
            self.mesh_beiya = self.mesh_beiya.simplify_quadric_decimation(8000)
        self.mesh_beiya = get_biggest_mesh(self.mesh_beiya)
        if self.AOI_or_UB == 0:
            self.insert_direction = get_insert_direction(self.mesh_beiya.as_open3d)
            print("get_insert_direction end")
        elif self.AOI_or_UB == 1:
            if self.insert_direction is None:
                self.insert_direction = get_insert_direction(self.mesh_beiya.as_open3d)
            self.mesh_beiya = filling_undercut(
                self.mesh_beiya.as_open3d, self.insert_direction
            )
            print("filling_undercut end")

    def trans_loop_geodesic(
        self,
        ng_points,
        points_id,
        points,
        max_iterations=10,
        condition_points=None,
        distance=0,
        mode=0,
    ):
        """
        使用测地距离进行迭代处理网格点的变换过程
        
        参数:
            ng_points: 需要处理的点索引数组
            points_id: 网格点的索引数组
            points: 网格点的坐标数组
            max_iterations: 最大迭代次数，默认为10
            condition_points: 条件点数组，默认为None（使用points）
            distance: 距离阈值，默认为0
            mode: 处理模式，0表示对侧处理，1表示背侧处理
            
        返回:
            ng_points: 处理后的点索引数组
            status: 状态码，1表示达到最大迭代次数，0表示正常完成
        """
        # 如果没有指定条件点，使用输入的点作为条件点
        if condition_points is None:
            condition_points = points
            
        iterations = 0
        # 将网格转换为Open3D格式
        mesh_o3d = self.mesh.as_open3d
        # 选择指定索引的点
        mesh_area = mesh_o3d.select_by_index(points_id)
        # 找到边界点和边界索引
        edge, edge_id = find_boundaries(mesh_area)
        
        # 创建图结构用于计算测地距离
        G = nx.Graph()
        # 添加所有顶点
        for v_index, v in enumerate(self.mesh.vertices):
            G.add_node(v_index, pos=v)
        # 添加所有边及其权重
        for edge in self.mesh.edges_unique:
            v1 = edge[0]
            v2 = edge[1]
            G.add_edge(v1, v2, weight=np.linalg.norm(self.mesh.vertices[v1] - self.mesh.vertices[v2]))
        
        # 迭代处理，直到满足条件或达到最大迭代次数
        while (
            np.intersect1d(ng_points, condition_points).size
            and iterations < max_iterations
        ):
            # 计算有符号距离的网格
            dis, _ = compute_signed_distance(
                self.mesh_beiya,
                self.mesh.vertices[ng_points],
            )
            ng_points_idx = np.where(dis > distance)[0]
            if len(ng_points_idx) == 0:
                break
            # 更新需要处理的点
            ng_points = ng_points[ng_points_idx]
            dis = dis[ng_points_idx]
            
            control_points = ng_points[np.argmax(dis)]
            source_points = self.mesh.vertices[control_points]
            target_points = source_points + (self.axis_direct / np.linalg.norm(self.axis_direct)) * dis[np.argmax(dis)]
            
            self.deform_mesh(control_points, target_points, ng_points)
            
            iterations += 1
            
            if iterations >= max_iterations:
                break
            
        # 返回处理结果和状态
        if iterations >= max_iterations:
            return ng_points, 1  # 达到最大迭代次数
        else:
            return ng_points, 0  # 正常完成

    def deform_mesh(self, control_point, target_point, affected_points):
        """
        基于测地距离的网格变形，并对受影响区域进行Laplacian平滑
        
        参数:
            control_point: 控制点的索引
            target_point: 控制点的目标位置
            affected_points: 受影响的点集索引数组
            
        返回:
            bool: 是否进行了有效的变形
        """
        # 创建图结构用于计算测地距离
        G = nx.Graph()
        # 添加所有顶点
        for v_index, v in enumerate(self.mesh.vertices):
            G.add_node(v_index, pos=v)
        # 添加所有边及其权重
        for edge in self.mesh.edges_unique:
            v1 = edge[0]
            v2 = edge[1]
            G.add_edge(v1, v2, weight=np.linalg.norm(self.mesh.vertices[v1] - self.mesh.vertices[v2]))
        
        # 计算到受影响点的测地距离
        distances = {}
        for v in affected_points:
            try:
                dist = nx.shortest_path_length(G, source=control_point, target=v, weight='weight')
                distances[v] = dist
            except nx.NetworkXNoPath:
                continue
        
        if not distances:
            return False
        
        # 使用最大距离作为影响半径
        max_dist = max(distances.values())
        
        # 检查最小权重是否超过阈值
        min_weight = 1 - (max(distances.values()) / max_dist)
        if min_weight > 0.9:
            return False
        
        # 计算权重并应用位移
        new_vertices = np.copy(self.mesh.vertices)
        control_displacement = target_point - self.mesh.vertices[control_point]
        
        # 计算位移
        for v, dist in distances.items():
            # 计算权重
            weight = 1 - (dist / max_dist)
            # 计算位移
            displacement = control_displacement.reshape(3,) * weight
            # 应用位移
            new_vertices[v] += displacement
        
        # 更新网格
        self.mesh.update_mesh(new_vertices)
        
        # 获取所有需要保留的点
        neighbors = get_neighbors(affected_points, 3, self.mesh)
        keep_points = np.unique([t for x in neighbors for t in x])
        
        # 获取需要删除的点
        all_points = np.arange(len(self.mesh.vertices))
        remove_points = np.setdiff1d(all_points, keep_points)
        
        # 保存原始点的位置
        original_positions = self.mesh.vertices[keep_points].copy()
        
        # 转换为Open3D格式并删除点
        mesh_o3d = self.mesh.as_open3d
        mesh_o3d.remove_vertices_by_index(remove_points)
        
        mesh_out = mesh_o3d.filter_smooth_laplacian(number_of_iterations=10)
        mesh_out.compute_vertex_normals()
        
        # 计算平滑后的位移
        displacements = np.asarray(mesh_out.vertices) - original_positions
        
        # 应用平滑后的位移到原始网格
        self.mesh.vertices[keep_points] += displacements
        
        # 更新网格
        self.mesh.update_mesh()
        
        return True


def stdcrown(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh1=event.get("mesh1"),
        mesh2=event.get("mesh2"),
        mesh_beiya=event.get("mesh_beiya"),
        mesh_upper=event.get("mesh_upper"),
        mesh_lower=event.get("mesh_lower"),
        kps=event.get("kps"),
        all_other_crowns=event.get("all_other_crowns"),
        miss_id=event.get("beiya_id"),
        voxel_logits=event.get("voxel_logits"),
        is_single=event.get("is_single"),
        pt1=event.get("pt1"),
        pt2=event.get("pt2"),
        new_transform_list=event.get("new_transform_list"),
        transform=event.get("transform"),
        ai_matrix=event.get("ai_matrix"),
        template_name=event.get("template_name", "st_tooth"),
        handler_name="std",
    )
    generate_crowns.get_std_crown()
    return generate_crowns


def post(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh1=read_mesh_bytes(event.get("closer")),
        mesh2=read_mesh_bytes(event.get("further")),
        mesh_beiya=read_mesh_bytes(event.get("inner")),
        mesh_upper=read_mesh_bytes(event.get("mesh_upper")),
        mesh_lower=read_mesh_bytes(event.get("mesh_lower")),
        miss_id=event.get("beiya_id"),
        is_single=event.get("is_single"),
        cpu_points_info=event.get("cpu_points_info"),
        cpu_colors_info=event.get("cpu_colors_info"),
        points_oppo_id=event.get("points_oppo_id"),
        paras=event.get("paras"),
        mesh=read_mesh_bytes(event.get("standard")),
        mesh_jaw=read_mesh_bytes(event.get("mesh_jaw")),
        mesh_oppo=read_mesh_bytes(event.get("mesh_oppo")),
        pre_tag=event.get("pre_tag"),
        trans_matrix=event.get("crown_rot_matirx", np.eye(4)),
        template_name=event.get("template_name", "st_tooth"),
        undercut_mesh=event.get("undercut_mesh"),
        points_info=event.get("points_info"),
        handler_name="post",
    )

    generate_crowns.get_post_mesh()

    return generate_crowns


def occlu(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh=read_mesh_bytes(event.get("out")),
        mesh_oppo=read_mesh_bytes(event.get("mesh_oppo")),
        mesh_beiya=read_mesh_bytes(event.get("inner")),
        paras=event.get("paras"),
        mesh1=read_mesh_bytes(event.get("closer")),
        mesh2=read_mesh_bytes(event.get("further")),
        miss_id=event.get("beiya_id"),
        is_single=event.get("is_single"),
        cpu_points_info=event.get("cpu_points_info"),
        axis=event.get("axis"),
        trans_matrix=event.get("trans_matrix", np.eye(4)),
        template_name=event.get("template_name", "st_tooth"),
        handler_name="occlu",
    )
    generate_crowns.get_occlu_mesh()
    return generate_crowns


def adjust(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh=event.get("out"),
        mesh_beiya=event.get("inner"),
        paras=event.get("paras"),
        handler_name="adjust",
    )
    generate_crowns.get_adjust_mesh()

    return generate_crowns


def stitch_edge(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh=read_mesh_bytes(event.get("out")),
        mesh_beiya=read_mesh_bytes(event.get("inner")),
        add_points=event.get("add_points"),
        miss_id=event.get("beiya_id"),
        axis=event.get("axis"),
        align_edges=event.get("align_edges", True),
        mesh_jaw=read_mesh_bytes(event.get("mesh_jaw")),
        handler_name="stitch",
    )
    generate_crowns.stitch_edge()

    return generate_crowns


def undercut_filling(event):
    generate_crowns = GenerateCrowns()
    generate_crowns.load_data(
        mesh_beiya=read_mesh_bytes(event.get("inner")),
        insert_direction=event.get("AOI"),
        AOI_or_UB=event.get("AOI_or_UB"),
        prep_extended=read_mesh_bytes(event.get("prep_extended")),
        handler_name="undercut",
    )
    generate_crowns.undercpt_filling()
    return generate_crowns


if __name__ == "__main__":
    pass
            