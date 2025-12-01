import base64
import copy
import random
import pylfda
import DracoPy
import MQCompressPy
import numpy as np
import open3d
import trimesh
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from tps import TPS
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
        quantization_bits=20,
        compression_level=10,
        colors=colors,
    )
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def compress_drc(mesh, points_id=[]):
    vert_flags = np.zeros(len(mesh.vertices), dtype=np.uint8)
    for i in range(len(points_id)):
        vert_flags[points_id[i]] = i + 1
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
    if buffer is not None or buffer != "":
        a = base64.b64decode(buffer)
        mesh_object = DracoPy.decode_buffer_to_mesh(a)
        V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
        F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
        return trimesh.Trimesh(V, F).as_open3d
    else:
        return None


def o3d2tri(mesh_o3d):
    return trimesh.Trimesh(mesh_o3d.vertices, mesh_o3d.triangles)


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
    if type(mesh) is not trimesh.Trimesh:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
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

    for i in range(10):
        if len(faces1) != len(faces2):
            idx = min([x for x in range(len(faces2)) if (faces1[x] != faces2[x]).all()])
            if len(faces1) > len(faces2):
                faces1 = np.vstack((faces1[:idx], faces1[idx + 1 :]))
            else:
                faces2 = np.vstack((faces2[:idx], faces2[idx + 1 :]))
        else:
            break
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


def angle_between_vectors_batch(a, b, ignore=-1):
    """
    计算两组向量之间的夹角（单位：弧度）

    参数:
    a -- 第一组向量，形状为 (n, 3)
    b -- 第二组向量，形状为 (n, 3)
    ignore -- 忽略的维度，-1 表示不忽略任何维度

    返回:
    角度数组，形状为 (n,)
    """
    if ignore != -1:
        a = a.copy()  # 避免修改原始数组
        b = b.copy()
        a[:, ignore] = b[:, ignore]

    dot_product = np.sum(a * b, axis=1)
    magnitude_a = np.linalg.norm(a, axis=1)
    magnitude_b = np.linalg.norm(b, axis=1)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 避免数值计算误差导致超出范围
    return angle_rad


class MeshCutter:
    def __init__(self, v, f):
        self.vertices = np.array(v)
        self.faces = np.array(f)
        self.vertex_tree = cKDTree(self.vertices)

        # 构建顶点邻接矩阵
        self.adjacency_matrix = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        """构建顶点邻接矩阵"""
        n_vertices = len(self.vertices)
        # 创建稀疏矩阵
        rows = []
        cols = []
        data = []

        # 遍历所有面
        for face in self.faces:
            # 添加面的三条边
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                # 计算边的长度
                edge_length = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                # 添加双向边
                rows.extend([v1, v2])
                cols.extend([v2, v1])
                data.extend([edge_length, edge_length])

        # 创建稀疏矩阵
        return csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

    def _find_nearest_vertex(self, point):
        """找到网格上距离给定点最近的顶点"""
        distance, vertex_idx = self.vertex_tree.query(point)
        return vertex_idx

    def _compute_geodesic_path(self, start_idx, end_idx):
        """计算两点之间的测地线路径"""
        # 使用Dijkstra算法计算最短路径
        distances, predecessors = dijkstra(
            self.adjacency_matrix,
            directed=False,
            indices=start_idx,
            return_predecessors=True,
        )

        # 重建路径
        path = []
        current = end_idx
        while current != start_idx:
            path.append(current)
            current = predecessors[current]
        path.append(start_idx)
        return path[::-1]  # 反转路径使其从起点到终点

    def cut_mesh(self, contour_points):
        """
        根据轮廓点切割网格
        :param contour_points: 轮廓点列表，每个点是一个3D坐标
        :return: 切割后的网格
        """
        # 找到轮廓点对应的最近顶点
        contour_vertices = [
            self._find_nearest_vertex(point) for point in contour_points
        ]

        # 计算测地线路径
        paths = []
        for i in range(len(contour_vertices)):
            start_idx = contour_vertices[i]
            end_idx = contour_vertices[(i + 1) % len(contour_vertices)]
            path = self._compute_geodesic_path(start_idx, end_idx)
            paths.extend(path)

        # 创建切割后的网格
        # 1. 标记要保留的面
        faces_to_keep = []
        for face in self.faces:
            # 检查面的所有顶点是否在切割路径上
            if not any(vertex in paths for vertex in face):
                faces_to_keep.append(face)

        # 2. 创建新的网格
        if faces_to_keep:
            new_mesh = trimesh.Trimesh(vertices=self.vertices, faces=faces_to_keep)
            return new_mesh
        else:
            return None


def find_boundaries(mesh):
    if type(mesh) is trimesh.Trimesh:
        mesh_o3d = mesh.as_open3d
    else:
        mesh_o3d = mesh
    a = mesh_o3d.get_non_manifold_edges(allow_boundary_edges=True)
    b = mesh_o3d.get_non_manifold_edges(allow_boundary_edges=False)
    a = np.unique(np.asarray(a).flatten())
    b = np.unique(np.asarray(b).flatten())
    out = b[~np.isin(b, a)]

    return np.asarray(mesh.vertices)[out], out


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


def get_thickness_gap(mesh_input):
    def find_unique_elements(lst):
        array = np.array(lst)
        unique, counts = np.unique(array, return_counts=True)
        unique_elements = unique[counts == 1]
        return unique_elements.tolist()

    mesh_inner = copy.deepcopy(mesh_input)
    # mesh_inner.invert()
    if mesh_inner.faces.shape[0] > 10000:
        # desired_count = 5000
        # mesh = pylfda.Mesh()
        # mesh.vertices, mesh.faces = mesh_inner.vertices, mesh_inner.faces
        # decimationType = pylfda.DecimationType.Vertex
        # max_normal_deviation = 1
        # fix_boundary = True
        # out = pylfda.decimate_mesh(
        #     mesh, desired_count, decimationType, max_normal_deviation, fix_boundary
        # )
        # if out:
        #     mesh_inner = trimesh.Trimesh(mesh.vertices, mesh.faces)
        mesh_inner_o3d = mesh_inner.as_open3d
        mesh_inner_o3d = mesh_inner_o3d.simplify_quadric_decimation(
            target_number_of_triangles=10000,
        )
        mesh_inner = trimesh.Trimesh(mesh_inner_o3d.vertices, mesh_inner_o3d.triangles)
    for _ in range(5):
        result = find_unique_elements(mesh_inner.faces.flatten())
        if not len(result):
            break
        mesh_inner_o3d = mesh_inner.as_open3d
        mesh_inner_o3d.remove_vertices_by_index(result)
        mesh_inner_o3d.compute_vertex_normals()
        mesh_inner = trimesh.Trimesh(
            np.round(mesh_inner_o3d.vertices, 4),
            mesh_inner_o3d.triangles,
            vertex_normals=mesh_inner_o3d.vertex_normals,
        )
    inner = copy.deepcopy(mesh_inner)
    mesh_inner = mesh_inner.as_open3d
    mesh_inner.compute_vertex_normals()
    mesh_inner = trimesh.Trimesh(mesh_inner.vertices, mesh_inner.triangles)
    points = []
    outlines = mesh_inner.outline().referenced_vertices
    # 构建 KD 树
    tree = cKDTree(mesh_inner.vertices[outlines])
    # 计算每个点到另一个点云的最小距离
    distances, v2edge_id = tree.query(mesh_inner.vertices)
    n = []
    o_n = []
    k_p = []
    for k in range(len(mesh_inner.vertices)):
        a = mesh_inner.vertex_normals[k]
        if k in outlines:
            o_n.append(k)
            k_p.append(k)
            points.append(mesh_inner.vertices[k])
            # points.append(mesh_inner.vertices[k])
            continue
        n_id = get_neighbors([k], 3, mesh_inner)
        b = mesh_inner.vertex_normals[np.unique([t for x in n_id for t in x])]
        cos_angle = [np.dot(a, x) for x in b]
        angle = np.max(np.arccos(np.array(cos_angle) - 1e-6) / 3.141592653 * 180)
        if distances[k] < 0.5:
            outlines = np.append(outlines, k)
            o_n.append(k)
            k_p.append(k)
            points.append(mesh_inner.vertices[k] + a * 0.01)
            continue
        # elif distances[k] < 0.5:
        #     o_n.append(k)
        #     k_p.append(k)
        #     # points.append(mesh_inner.vertices[k] + a * (0.1 / 0.4 * (distances[k] - 0.1)))
        #     points.append(mesh_inner.vertices[k] + a * 0.01)
        #     continue
        elif distances[k] < 1:
            # o_n.append(k)
            if angle > 20:
                n.append(k)
                points.append(mesh_inner.vertices[k])
                continue
            else:
                points.append(
                    mesh_inner.vertices[k]
                    + a * (0.6 - 0.01) / 1.5 * (distances[k] - 0.5)
                )
                continue
        else:
            if angle > 20:
                n.append(k)
                points.append(mesh_inner.vertices[k])
                continue
            else:
                v_n = a
                points.append(mesh_inner.vertices[k] + v_n * 0.55)
    points_id = [x for x in range(len(mesh_inner.vertices)) if x not in k_p]
    if len(points_id) > len(mesh_inner.vertices) // 3:
        points_id = random.sample(
            points_id,
            len(mesh_inner.vertices) // 3,
        )
    points_id = [x for x in points_id if x not in n]
    points_id.extend(k_p)
    points_tps = np.array(points)[points_id]
    trans = TPS(mesh_inner.vertices[points_id], points_tps, lambda_=0.5)
    mesh_beiya = trimesh.Trimesh(trans(mesh_inner.vertices), mesh_inner.faces)
    # mesh_beiya = tps(mesh_inner, points_id, points_tps)
    dis_p = [x for x in range(len(mesh_inner.vertices)) if x not in o_n]
    dis = compute_signed_distance(mesh_inner, mesh_beiya.vertices[dis_p])[0]
    dis_id = np.where(dis > 0)[0]
    if len(dis_id):
        p_id = np.array(dis_p)[dis_id]
        p = mesh_beiya.vertices[p_id]
        p += mesh_beiya.vertex_normals[p_id] * (dis[dis_id] + 0.55).reshape(-1, 1)
        p_id = np.hstack([p_id, o_n])
        p = np.vstack([p, mesh_beiya.vertices[o_n]])
        not_in_p_id = [x for x in range(len(mesh_beiya.vertices)) if x not in p_id]
        n_p_id = random.sample(not_in_p_id, len(not_in_p_id) // 5)
        p_id = np.hstack([p_id, n_p_id])
        p = np.vstack([p, mesh_beiya.vertices[n_p_id]])
        trans_ = TPS(mesh_beiya.vertices[p_id], p, lambda_=0.5)
        thickness_shell = trimesh.Trimesh(trans_(mesh_beiya.vertices), mesh_beiya.faces)
        # thickness_shell = tps(mesh_beiya, p_id, p)
    else:
        thickness_shell = mesh_beiya
    thickness_shell.vertices[outlines] = np.array(points)[outlines]

    t_o3d = thickness_shell.as_open3d
    t_o3d.remove_vertices_by_index(outlines)
    thickness_shell = trimesh.Trimesh(t_o3d.vertices, t_o3d.triangles)

    return thickness_shell, inner


def get_thickness_gap2(thickness_shell, mesh_crown):
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh_crown))
    in_mesh = scene.compute_occupancy(
        np.array(thickness_shell.vertices, dtype=np.float32)
    )
    dis_id = np.where(in_mesh.numpy() == 0)[0]
    if len(dis_id):
        p = scene.compute_closest_points(
            np.array(thickness_shell.vertices[dis_id], dtype=np.float32)
        )["points"].numpy()
        thickness_shell.vertices[dis_id] = thickness_shell.vertices[dis_id] + 1.3 * (
            p - thickness_shell.vertices[dis_id]
        )
    return thickness_shell


def sort_and_remove_close_points(points, threshold=0.5):
    """
    Sorts a set of 3D points that form an irregular closed curve and removes points that are too close to each other.
    Returns the indices of the sorted points.

    Parameters
    ----------
    points : (n, 3) float
        Array of 3D points.
    threshold : float, optional
        Minimum distance between points (default is 0.5).

    Returns
    -------
    sorted_indices : (m,) int
        Indices of the sorted points with close points removed.
    """
    # Initialize the sorted indices list with the index of the first point
    sorted_indices = [0]
    remaining_indices = list(range(1, len(points)))

    while remaining_indices:
        # Get the last point in the sorted list
        last_point = points[sorted_indices[-1]]

        # Calculate distances from the last point to all remaining points
        remaining_points = points[remaining_indices]
        distances = np.linalg.norm(remaining_points - last_point, axis=1)

        # Find the index of the closest point
        closest_index = np.argmin(distances)
        closest_distance = distances[closest_index]

        # Check if the closest point is too close
        if closest_distance > threshold:
            # Add the index of the closest point to the sorted list
            sorted_indices.append(remaining_indices[closest_index])
            # Remove the closest point from the remaining indices
            del remaining_indices[closest_index]
        else:
            # If the closest point is too close, remove it and continue
            del remaining_indices[closest_index]

    return np.array(sorted_indices)


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


def get_roi_from_upper_lower(mesh_jaw, mesh_beiya):
    mesh_jaw = get_roi(mesh_jaw, mesh_beiya.centroid, 10)
    mesh_jaw = mesh_jaw.simplify_quadric_decimation(25000)
    mesh_jaw = get_biggest_mesh(mesh_jaw)
    return mesh_jaw


def get_insert_direction(prep_mesh: open3d.geometry.TriangleMesh) -> np.ndarray:
    """Visulize the computed minimum-undercut direction of a prep tooth"""
    # 连接准备网格和邻接网格
    prep_mesh.compute_vertex_normals()
    prep_mesh_o3d_t = open3d.t.geometry.TriangleMesh.from_legacy(prep_mesh)
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(prep_mesh_o3d_t)  # 将网格添加到光线投射场景中

    # 球体的上半部分的所有点，用来和原点连接产生光线方向向量
    sphere = open3d.geometry.TriangleMesh.create_sphere(resolution=50)
    points = np.asarray(sphere.vertices)
    points = points[points[:, 1] > 0.8]
    # 找到 prep_mesh 的 y 值最高点
    max_y_prep_mesh = np.asarray(prep_mesh.vertices)[:, 1].max()
    # 生成均匀的点阵作为光线起点
    grid_size = 150  # 点阵大小，可以根据需要调整
    x_range = np.linspace(
        prep_mesh.get_max_bound()[0] * 2, prep_mesh.get_min_bound()[0] * 2, grid_size
    )
    z_range = np.linspace(
        prep_mesh.get_max_bound()[2] * 2, prep_mesh.get_min_bound()[2] * 2, grid_size
    )
    x_grid, z_grid = np.meshgrid(x_range, z_range)
    y_grid = np.full(
        x_grid.shape, max_y_prep_mesh + 1
    )  # y 坐标高于 prep_mesh 的 y 值最高点
    ray_origins = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    out_undercut_num = []
    hit_num = []
    # 遍历每个点进行光线交集测试
    print(len(points))
    for i, p in enumerate(points):
        ray_directions = np.array([p]).repeat(
            len(ray_origins), axis=0
        )  # 为所有起点重复方向向量
        rays = np.concatenate(
            (ray_origins, ray_directions * -1), axis=1
        )  # 组合起点和方向
        ray_intersections = scene.count_intersections(
            open3d.core.Tensor(rays.astype(np.float32))
        )  # 测试遮挡
        undercut_num = len(np.where(ray_intersections.numpy() > 1)[0])
        hit_num.append(len(np.where(ray_intersections.numpy() > 0)[0]))
        out_undercut_num.append(undercut_num)
    # 倒凹最小的前1/4个方向的平均值
    min_indices_num = np.argsort(out_undercut_num)[: int(points.shape[0] / 4)]
    out_direction = np.mean(points[min_indices_num], axis=0)[np.newaxis, ...]

    if out_direction[0][1] > 0:
        out_direction[0] *= -1.0
    return out_direction[0]


def getBoundaryPoints(mesh):
    """get the boundary points of a mesh IN THE ORDER of how they appear in the mesh"""
    if type(mesh) is not trimesh.Trimesh:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
    unique_elements, counts = np.unique(mesh.edges_sorted, return_counts=True, axis=0)
    edges = unique_elements[np.where(counts == 1)[0]]
    boundary_point_id = np.unique(edges.flatten())
    boundary_point = mesh.vertices[boundary_point_id]
    return boundary_point, boundary_point_id


def arrange_edges(edges):
    edges = edges.tolist()
    result = [edges.pop(0)]  # 从第一条边开始

    while edges:
        last = result[-1][1]  # 当前路径的末端
        for i, (u, v) in enumerate(edges):
            if u == last:
                result.append(edges.pop(i))
                break
            elif v == last:
                result.append([v, u])  # 反向连接
                edges.pop(i)
                break
        else:
            break  # 找不到连接的边，退出

    return np.array(result)


def boundarySmooth(mesh):
    if type(mesh) is not trimesh.Trimesh:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
    unique_elements, counts = np.unique(mesh.edges_sorted, return_counts=True, axis=0)
    edges = unique_elements[np.where(counts == 1)[0]]
    sorted_edges = arrange_edges(edges)
    v = np.array(mesh.vertices)
    for i in range(len(sorted_edges) - 1):
        p0 = mesh.vertices[sorted_edges[i]][0]
        p1 = mesh.vertices[sorted_edges[i]][1]
        p2 = mesh.vertices[sorted_edges[i + 1]][1]
        v[sorted_edges[i, 1]] = 0.5 * (0.5 * (p0 + p2) + p1)
    return trimesh.Trimesh(v, mesh.faces, process=False)


def compute_distances(A, B):
    distances_0 = np.sqrt(
        np.sum((A[:, np.newaxis, :1, :] - B[np.newaxis, :, :, :]) ** 2, axis=3)
    )
    distance_sums_0 = np.sum(distances_0, axis=2)
    closest_indices_0 = np.argmin(distance_sums_0, axis=1)
    distances_1 = np.sqrt(
        np.sum((A[:, np.newaxis, 1:, :] - B[np.newaxis, :, :, :]) ** 2, axis=3)
    )
    distance_sums_1 = np.sum(distances_1, axis=2)
    closest_indices_1 = np.argmin(distance_sums_1, axis=1)
    return np.column_stack((closest_indices_0, closest_indices_1))


def stitch_(mesh_to_stitch, mesh_before_stitch, edge_after, type="inner"):
    """
    mesh_to_stitch: 待缝合的mesh
    mesh_before_stitch: 缝合前的mesh
    edge_after: mesh_after_o3d的边缘索引 (n, 2) int
    type: "inner" or "outer", default is "inner"
    """
    while True:
        mesh_to_stitch_bound, mesh_to_stitch_bound_id = getBoundaryPoints(
            mesh_to_stitch
        )
        mesh_to_stitch_trimesh = trimesh.Trimesh(mesh_to_stitch.vertices, mesh_to_stitch.triangles)
        mask = np.isin(mesh_to_stitch_trimesh.edges_unique, mesh_to_stitch_bound_id).all(axis=1) 
        bad_points_info = np.unique(mesh_to_stitch_trimesh.edges_unique[mask], return_counts=True)
        bad_points = bad_points_info[0][np.where(bad_points_info[1] >= 4)[0]]
        if len(bad_points):
            mesh_to_stitch.remove_vertices_by_index(bad_points)
            mesh_to_stitch = get_biggest_mesh(mesh_to_stitch).as_open3d
            mesh_to_stitch_bound, mesh_to_stitch_bound_id = getBoundaryPoints(
                mesh_to_stitch
            )
            mesh_to_stitch_trimesh = trimesh.Trimesh(mesh_to_stitch.vertices, mesh_to_stitch.triangles)
        mesh_to_stitch_sort_idx = sort_and_remove_close_points(mesh_to_stitch_bound, 0)
        mesh_to_stitch_bound = mesh_to_stitch_bound[mesh_to_stitch_sort_idx]
        mesh_to_stitch_bound_id = mesh_to_stitch_bound_id[mesh_to_stitch_sort_idx]
        mesh_to_stitch_edge = [
            [mesh_to_stitch_bound_id[x], mesh_to_stitch_bound_id[x + 1]]
            if x < len(mesh_to_stitch_bound_id) - 1
            else [mesh_to_stitch_bound_id[x], mesh_to_stitch_bound_id[0]]
            for x in range(len(mesh_to_stitch_bound_id))
        ]

        mask = (mesh_to_stitch_trimesh.edges[:, None] == np.array(mesh_to_stitch_edge)[:, ::-1]).all(axis=2).any(axis=0) 
        if np.sum(mask) < len(mask) // 2:
            mask = (mesh_to_stitch_trimesh.edges[:, None] == np.array(mesh_to_stitch_edge)).all(axis=2).any(axis=0) 
        counts_a = np.unique(np.array(mesh_to_stitch_edge)[~mask].flatten(), return_counts=True)
        counts_b = np.unique(mesh_to_stitch_trimesh.faces, return_counts=True)
        bad_points = counts_a[0][np.where(counts_b[1][counts_a[0]] < 3)[0]]
        

        dist_index = compute_distances(
            np.asarray(mesh_to_stitch.vertices)[mesh_to_stitch_edge],
            mesh_before_stitch.vertices[edge_after],
        )

        if len(bad_points):
            mesh_to_stitch.remove_vertices_by_index(bad_points)
            mesh_to_stitch = get_biggest_mesh(mesh_to_stitch).as_open3d
            continue
        # else:
        #     break

        bad_bound_idx_0 = [
            mesh_to_stitch_edge[id][1]
            for id, x in enumerate(dist_index)
            if x[0] > x[1] and x[0] != np.max(dist_index)
        ]
        bad_bound_idx_1 = [
            mesh_to_stitch_edge[id][1]
            for id, x in enumerate(dist_index)
            if x[0] < x[1] and x[1] != np.max(dist_index)
        ]
        if len(bad_bound_idx_0) > len(bad_bound_idx_1):
            bad_bound_idx = bad_bound_idx_1
            dist_index = dist_index[::-1, ::-1]
            mesh_to_stitch_edge = np.array(mesh_to_stitch_edge)[::-1, ::-1]
        else:
            bad_bound_idx = bad_bound_idx_0
        if len(bad_bound_idx) == 0:
            break
        mesh_to_stitch.remove_vertices_by_index(bad_bound_idx)
        mesh_to_stitch = get_biggest_mesh(mesh_to_stitch).as_open3d

    mesh_to_stitch = trimesh.Trimesh(
        vertices=mesh_to_stitch.vertices, faces=mesh_to_stitch.triangles
    )

    new_faces = []
    used_egges = []
    for edge_idx, (i_0, i_1) in enumerate(dist_index):
        p0 = mesh_to_stitch_edge[edge_idx][0] + len(mesh_before_stitch.vertices)
        p1 = mesh_to_stitch_edge[edge_idx][1] + len(mesh_before_stitch.vertices)
        b_0_0 = edge_after[i_0][0]
        b_0_1 = edge_after[i_0][1]
        if i_0 != i_1:
            if (i_0 - i_1) > np.max(len(edge_after)) // 2:
                i_list = [x for x in range(i_0, len(edge_after), 1)]
                i_list.extend([x for x in range(0, i_1 + 1, 1)])
            else:
                i_list = [x for x in range(i_0 + 1, i_1 + 1, 1)]
            mask = np.isin(i_list, used_egges, invert=True)
            i_list = np.array(i_list)[mask]
            if len(i_list) == 0:
                new_faces.append([p1, p0, b_0_1])
                continue
            # outer_points = mesh_before_stitch.vertices[edge_after[i_list]]
            outer_points = mesh_before_stitch.vertices[np.unique(edge_after[i_list])]
            p0_point = mesh_to_stitch.vertices[p0 - len(mesh_before_stitch.vertices)]
            p1_point = mesh_to_stitch.vertices[p1 - len(mesh_before_stitch.vertices)]
            # dis_p0 = np.linalg.norm(p0_point - outer_points, axis=2)
            # dis_p1 = np.linalg.norm(p1_point - outer_points, axis=2)
            dis_p0 = np.linalg.norm(p0_point - outer_points, axis=1)
            dis_p1 = np.linalg.norm(p1_point - outer_points, axis=1)
            min_idx = np.argmin(dis_p0 + dis_p1)
            for p_id, list_id in enumerate(i_list):
                b0 = edge_after[list_id][0]
                b1 = edge_after[list_id][1]
                if p_id < min_idx:
                    new_faces.append([p0, b0, b1])
                elif p_id >= min_idx:
                    new_faces.append([p1, b0, b1])
                used_egges.append(list_id)
            if min_idx == len(i_list):
                new_faces.append([p1, p0, edge_after[i_list[-1]][1]])
            else:
                new_faces.append([p1, p0, edge_after[i_list[min_idx]][0]])
        else:
            if edge_idx == len(dist_index) - 1:
                if used_egges[0] == i_0:
                    new_faces.append([p1, p0, b_0_0])
                else:
                    new_faces.append([p1, p0, b_0_1])
            elif i_0 == np.max(edge_after):
                new_faces.append([p1, p0, b_0_0])
            else:
                new_faces.append([p1, p0, b_0_1])

        print
    if type == "outer":
        new_faces = np.array(new_faces)
        new_faces[:, :2] = new_faces[:, :2][:, ::-1]
    model_stitched = trimesh.Trimesh(
        np.concatenate((mesh_before_stitch.vertices, mesh_to_stitch.vertices), axis=0),
        np.concatenate(
            (
                mesh_before_stitch.faces,
                mesh_to_stitch.faces + len(mesh_before_stitch.vertices),
                new_faces,
            ),
            axis=0,
        ),
        process=False,
    )
    return model_stitched, mesh_to_stitch_bound_id + len(mesh_before_stitch.vertices)


def stitch_edge(mesh_out, mesh_inner):
    bound_out_ori, bound_out_id_ori = getBoundaryPoints(mesh_out)
    bound_out_sort_idx = sort_and_remove_close_points(bound_out_ori, 0.05)
    bound_out = bound_out_ori[bound_out_sort_idx]
    bound_out_id = bound_out_id_ori[bound_out_sort_idx]
    for i in range(len(bound_out_id) - 1):
        bound_face_id = np.intersect1d(
            np.where(mesh_out.faces == bound_out_id[i])[0],
            np.where(mesh_out.faces == bound_out_id[i + 1])[0],
        )
        if len(bound_face_id):
            bound_face = mesh_out.faces[bound_face_id[0]]
            b_p_0 = np.where(bound_face == bound_out_id[i])[0]
            b_p_1 = np.where(bound_face == bound_out_id[i + 1])[0]
            if (
                (b_p_0, b_p_1) == (0, 1)
                or (b_p_0, b_p_1) == (1, 2)
                or (b_p_0, b_p_1) == (2, 0)
            ):
                bound_out = bound_out[::-1]
                bound_out_id = bound_out_id[::-1]
            break

    p, dis, triangles = trimesh.base.proximity.closest_point(mesh_inner, bound_out)

    array = (bound_out - p) / 4
    p = p[::-1]
    for n in range(3, 0, -1):
        p_ = bound_out - array * n
        p_ = p_[::-1]
        p = np.concatenate((p, p_), axis=0)
    p = p[::-1]
    edge_inner_idx = []
    edge_outer_idx = []
    new_faces = []
    new_bound_out_id = np.arange(len(bound_out_id))
    for i in range(len(bound_out_id)):
        p0_id = len(bound_out_id) + i
        p1_id = len(bound_out_id) + (i + 1 if i + 1 < len(bound_out_id) else 0)
        b0_id = new_bound_out_id[i]
        b1_id = new_bound_out_id[i + 1 if i + 1 < len(bound_out_id) else 0]
        new_faces.append([b0_id, b1_id, p1_id])
        new_faces.append([p0_id, b0_id, p1_id])
        edge_outer_idx.append([b0_id, b1_id])
        for j in range(3):
            new_faces.append(
                [
                    p0_id + j * len(bound_out),
                    p1_id + j * len(bound_out),
                    p1_id + (j + 1) * len(bound_out),
                ]
            )
            new_faces.append(
                [
                    p0_id + (j + 1) * len(bound_out),
                    p0_id + j * len(bound_out),
                    p1_id + (j + 1) * len(bound_out),
                ]
            )
            if j == 2:
                edge_inner_idx.append(
                    [
                        p0_id + (j + 1) * len(bound_out),
                        p1_id + (j + 1) * len(bound_out),
                    ]
                )

    edge_inner_idx = np.array(edge_inner_idx)
    edge_outer_idx = np.array(edge_outer_idx)

    model_bound = trimesh.Trimesh(
        np.concatenate((mesh_out.vertices[bound_out_id], p), axis=0),
        np.array(new_faces),
        process=False,
    )

    bound_points_idx = np.arange(len(model_bound.vertices))
    # bound_inner_idx = np.unique(edge_inner_idx)
    bound_outer_idx = np.unique(edge_outer_idx)
    bound_inner_idx = np.setdiff1d(bound_points_idx, bound_outer_idx)

    distance_bound_inner = np.linalg.norm(
        model_bound.vertices[np.unique(edge_inner_idx)][:, np.newaxis, :]
        - mesh_inner.vertices,
        axis=2,
    )
    remove_points_idx = np.unique(np.where(distance_bound_inner < 0.1)[1])
    mesh_inner_o3d = mesh_inner.as_open3d
    mesh_inner_o3d.remove_vertices_by_index(remove_points_idx)

    unique_elements, counts = np.unique(
        np.asarray(mesh_inner_o3d.triangles), return_counts=True
    )
    mesh_inner_o3d.remove_vertices_by_index(unique_elements[np.where(counts == 1)[0]])
    mesh_inner_o3d.remove_duplicated_vertices()
    mesh_inner_o3d.remove_unreferenced_vertices()
    mesh_inner_o3d = boundarySmooth(mesh_inner_o3d).as_open3d
    model_bound_inner, bounds_inner_self_idx = stitch_(
        mesh_inner_o3d, model_bound, edge_inner_idx
    )
    inner_points_idx = np.arange(
        len(model_bound.vertices), len(model_bound_inner.vertices)
    )

    neighbor = get_neighbors(bound_out_id, 3, mesh_out)
    neighbor = np.unique([t for x in neighbor for t in x])
    distance = np.linalg.norm(
        model_bound.vertices[np.unique(edge_outer_idx)][:, np.newaxis, :]
        - mesh_out.vertices[neighbor],
        axis=2,
    )
    remove_points_idx = np.unique(np.where(distance < 0.1)[1])
    mesh_outer_o3d = mesh_out.as_open3d
    mesh_outer_o3d.remove_vertices_by_index(np.array(neighbor)[remove_points_idx])

    unique_elements, counts = np.unique(
        np.asarray(mesh_outer_o3d.triangles), return_counts=True
    )
    mesh_outer_o3d.remove_vertices_by_index(unique_elements[np.where(counts == 1)[0]])
    mesh_outer_o3d = get_biggest_mesh(mesh_outer_o3d).as_open3d
    mesh_outer_o3d.remove_duplicated_vertices()
    mesh_outer_o3d.remove_unreferenced_vertices()
    mesh_outer_o3d.remove_degenerate_triangles()
    mesh_outer_o3d = boundarySmooth(mesh_outer_o3d).as_open3d

    model_bound_outer, bounds_outer_self_idx = stitch_(
        mesh_outer_o3d, model_bound_inner, edge_outer_idx, type="outer"
    )
    outer_points_idx = np.arange(
        len(model_bound_inner.vertices), len(model_bound_outer.vertices)
    )

    v = np.array(model_bound_outer.vertices)

    smoothed_points_idx = np.concatenate(
        [bound_points_idx, bounds_inner_self_idx, bounds_outer_self_idx]
    )
    faces = model_bound_outer.faces
    f_mask = np.all(np.isin(faces, smoothed_points_idx), axis=1)
    b = np.zeros(np.max(smoothed_points_idx) + 1, dtype=np.int32)
    a = np.vstack([smoothed_points_idx, np.arange(len(smoothed_points_idx))])
    b[a[0]] = a[1]

    mesh_bound_outer_removed = trimesh.Trimesh(
        model_bound_outer.vertices[smoothed_points_idx],
        b[faces[f_mask]],
        process=False,
    )
    model_laplacian = trimesh.smoothing.filter_taubin(
        mesh_bound_outer_removed,
    )
    v[smoothed_points_idx[: len(bound_points_idx)]] = model_laplacian.vertices[
        : len(bound_points_idx)
    ]
    model_laplacian_edge = trimesh.Trimesh(
        vertices=v,
        faces=model_bound_outer.faces,
        process=False,
    )
    model_laplacian_edge.fix_normals()
    return (
        model_laplacian_edge,
        outer_points_idx,
        inner_points_idx,
        bound_outer_idx,
        bound_inner_idx,
    )


if "__main__" == __name__:
    import os
    import json
    import traceback
    import pylfda

    mesh_out = trimesh.load('test_data/c2/15_stitch_fixed_inlay_outer_47.ply')
    mesh_inner = trimesh.load('test_data/c2/15.1_stitched_inlay_inner_47.ply')
    out = stitch_edge(mesh_out, mesh_inner)[0]
    out.export('test_data/c2/out.stl') 