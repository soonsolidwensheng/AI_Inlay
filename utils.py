import base64

import DracoPy
import MQCompressPy
import numpy as np
import open3d
import trimesh
import networkx as nx

# import pypruners


def read_mesh(path: str) -> open3d.geometry.TriangleMesh:
    mesh = open3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def write_mesh_bytes(mesh, preserve_order=False, colors=None):
    # 设置 Draco 编码选项
    encoding_test = DracoPy.encode_mesh_to_buffer(
        mesh.vertices,
        mesh.faces,
        preserve_order=preserve_order,
        quantization_bits=14,
        compression_level=10,
        colors=colors,
    )
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def compress_drc(mesh, points_id=[]):
    vert_flags = np.zeros(len(mesh.vertices), dtype=np.uint8)
    vert_flags[points_id] = 1
    # trimesh.PointCloud(mesh.vertices[points_id]).export('p.ply')
    in_mesh = MQCompressPy.MQC_Mesh()
    in_mesh.verts = MQCompressPy.VerticeArray(mesh.vertices)
    in_mesh.faces = MQCompressPy.FaceArray(mesh.faces)
    in_vert_flags = MQCompressPy.VerticeFlag_UINT8(
        np.array(vert_flags).astype(np.uint8)
    )
    compressed_data, error_code = MQCompressPy.compressMesh_UINT8(
        in_mesh, in_vert_flags
    )
    # with open('mesh.drc', 'wb') as f:
    #     f.write(compressed_data)
    if error_code == 0:
        b64_bytes = base64.b64encode(compressed_data)
        b64_str = b64_bytes.decode("utf-8")
        return b64_str
    else:
        assert "drc compress error"


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F).as_open3d


def compute_signed_distance(mesh: trimesh.Trimesh, q_points):
    mesh_o3d = open3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d)
    scene = open3d.t.geometry.RaycastingScene()
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


def get_biggest_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    获取备牙里最大的连通区
    """
    mesh_split = mesh.split(only_watertight=False)
    mesh = mesh_split[np.argmax(np.array([x.vertices.shape[0] for x in mesh_split]))]
    return mesh


# def pruner_fun(mesh: trimesh.Trimesh, interest_verts: np.ndarray, max_distance=1.0, min_distance=-2.5, max_angle=1.5) -> trimesh.Trimesh:
#     """
#     通过一组点对网格进行修剪。
#     参数:
#     mesh (trimesh.Trimesh): 输入的网格。
#     interest_verts (numpy.ndarray): 用于修剪的点。
#     max_distance (float): 修剪的最大距离。
#     min_distance (float): 修剪的最小距离。
#     max_angle (float): 修剪的最大角度。
#     返回:
#     mesh (trimesh.Trimesh): 修剪后的网格。
#     """
#     interest_verts = mesh.nearest.vertex(interest_verts)[1].reshape(-1, 1)
#     mesh_out = pypruners.pruner(
#         mesh.vertices, mesh.faces, interest_verts, max_distance, min_distance, max_angle
#     )
#     return mesh_out


def compute_centroid(vertices):
    """计算三角形顶点的质心"""
    return np.mean(vertices, axis=0)


def find_changed_faces(mesh1, mesh2, threshold=0.05):
    vertices1 = np.array(mesh1.vertices)
    faces1 = np.array(mesh1.faces)

    vertices2 = np.array(mesh2.vertices)
    faces2 = np.array(mesh2.faces)

    # 确保两个网格的面数量相同
    assert faces1.shape == faces2.shape, "两个网格的面数量不一致"

    # 计算每个面的质心
    centroids1 = np.array([compute_centroid(vertices1[face]) for face in faces1])
    centroids2 = np.array([compute_centroid(vertices2[face]) for face in faces2])

    # 计算每个面的法向量
    # normals1 = compute_face_normals(vertices1, faces1)
    # normals2 = compute_face_normals(vertices2, faces2)

    # 计算质心和法向量的差异矩阵
    centroid_diffs = np.linalg.norm(centroids1 - centroids2, axis=1)
    # normal_diffs = np.linalg.norm(normals1 - normals2, axis=1)

    # 使用布尔索引找到发生变化的面
    # changed_faces = np.where((centroid_diffs > threshold) | (normal_diffs > threshold))[
    #     0
    # ]
    changed_faces = np.where(centroid_diffs > threshold)[0]

    return changed_faces

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