import numpy as np
import trimesh
import time
from scipy.spatial import cKDTree


def compute_centroid(vertices):
    """计算三角形顶点的质心"""
    return np.mean(vertices, axis=0)


def compute_centroids(vertices, faces):
    """计算面片顶点的质心"""
    centroids = np.mean(vertices[faces], axis=1)
    return centroids


def project_points_to_plane(points, plane_point, plane_normal):
    """将点投影到平面上"""

    d = np.dot(plane_normal, plane_point)
    t = (d - np.dot(points, plane_normal[:, np.newaxis])).flatten() / np.dot(
        plane_normal, plane_normal
    )
    projections = points + t[:, np.newaxis] * plane_normal
    return projections


def is_point_in_triangle_batch(points, v1, v2, v3):
    """批量检查点是否在三角形内"""

    def sign(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
            p1[:, 1] - p3[1]
        )

    b1 = sign(points, v1, v2) < 0.0
    b2 = sign(points, v2, v3) < 0.0
    b3 = sign(points, v3, v1) < 0.0

    return (b1 == b2) & (b2 == b3)


def find_subdivided_faces(
    original_vertices,
    original_faces,
    subdivided_vertices,
    subdivided_faces,
):
    original_to_subdivided = []
    # p = []

    # 计算细分后面片的质心
    subdivided_centroids = compute_centroids(subdivided_vertices, subdivided_faces)

    # 构建 k-d 树以加速邻近查找
    kdtree = cKDTree(subdivided_centroids)

    for original_face in original_faces:
        v1, v2, v3 = original_vertices[original_face]
        plane_point = v1
        plane_normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(plane_normal)
        if norm == 0:
            continue
        plane_normal = plane_normal / norm
        original_centroid = compute_centroid([v1, v2, v3])
        max_dis = np.max(
            [
                np.linalg.norm(original_centroid - v1),
                np.linalg.norm(original_centroid - v2),
                np.linalg.norm(original_centroid - v3),
            ]
        )

        # 使用 k-d 树查找与原始面片质心距离在 max_dis 范围内的细分面片
        indices = kdtree.query_ball_point(original_centroid, max_dis)
        filtered_faces = subdivided_faces[indices]
        filtered_centroids = subdivided_centroids[indices]

        # 将筛选出的质心投影到原始面片的平面上
        projected_centroids = project_points_to_plane(
            filtered_centroids, plane_point, plane_normal
        )

        # 批量检查投影点是否在原始面片内
        in_triangle_mask = is_point_in_triangle_batch(projected_centroids, v1, v2, v3)
        corresponding_faces = filtered_faces[in_triangle_mask]
        # p.append(filtered_centroids[in_triangle_mask])

        original_to_subdivided.append(corresponding_faces)

    return original_to_subdivided


def compute_face_normals(vertices, faces):
    """计算每个面的法向量"""
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    normals = np.cross(v2 - v1, v3 - v1)
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals


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
    changed_faces = np.where(centroid_diffs > threshold)[
        0
    ]

    return changed_faces



# original_mesh = trimesh.load("ori.stl")

# # 示例数据
# original_vertices = np.array(original_mesh.vertices)

# original_faces = original_mesh.faces[np.load("face_id.npz.npy")]

# subdivided_mesh = trimesh.load("sub.stl")
# # 假设这是细分后的数据
# subdivided_vertices = np.array(subdivided_mesh.vertices)

# subdivided_faces = np.array(subdivided_mesh.faces)

# s1 = time.time()
# # 找到原始面片在细分后的面片集合
# original_to_subdivided = find_subdivided_faces(
#     original_vertices, original_faces, subdivided_vertices, subdivided_faces
# )
# s2 = time.time()
# print(s2 - s1)
# trimesh.PointCloud(
#     subdivided_mesh.vertices[
#         np.unique(np.array([z for x in original_to_subdivided for y in x for z in y]))
#     ]
# ).export("1.ply")
# trimesh.PointCloud(
#     original_mesh.vertices[np.unique(original_faces.reshape(-1))]
# ).export("2.ply")
