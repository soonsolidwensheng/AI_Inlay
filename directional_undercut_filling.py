import open3d as o3d
import numpy as np
from tps import tps
import time
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import r2_score, mean_squared_error

def align_mesh_to_direction(mesh, direction):
    # 计算旋转矩阵，使得 direction 与 y 轴负方向对齐
    target_direction = np.array([0, -1, 0])
    direction = direction / np.linalg.norm(direction)
    rotation_vector = np.cross(direction, target_direction)
    rotation_angle = np.arccos(np.dot(direction, target_direction))
    rotation_matrix = R.from_rotvec(rotation_vector * rotation_angle).as_matrix()

    # 变换网格顶点
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = vertices @ rotation_matrix.T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh, rotation_matrix

def inverse_transform_mesh(mesh, rotation_matrix):
    # 逆变换网格顶点
    vertices = np.asarray(mesh.vertices)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    transformed_vertices = vertices @ inverse_rotation_matrix.T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh

def find_blocking_vertices(mesh, direction=[0, -1, 0], epsilon=1e-6):
    """
    Find each vertex that the ray starting from it along the direction penetrates into the mesh.
    """
    # Convert to TensorTriangleMesh for RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    # Extract vertices
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    blocking_vertices, intersection_vertices, normals = [], [], []

    direction_np = np.array(direction)

    for vid, v in enumerate(vertices):
        ray = np.concatenate([v, direction_np], axis=0)

        # Cast rays and get hit distances
        rays_tensor = o3d.core.Tensor([ray], dtype=o3d.core.Dtype.Float32)
        intersections = scene.list_intersections(rays_tensor)
        t_hit = intersections['t_hit'].numpy()
        primitive_ids = intersections['primitive_ids'].numpy()

        # Find the second hit
        if len(t_hit) >= 2 and t_hit[-1] > epsilon:
            blocking_vertices.append(v)
            index_num = 0
            for triangle_id in primitive_ids:
                if vid not in triangles[triangle_id]:
                    tri_verts = triangles[triangle_id]
                    v0 = vertices[tri_verts[0]]
                    v1 = vertices[tri_verts[1]]
                    v2 = vertices[tri_verts[2]]
                    #计算穿透点
                    uu, vv = intersections['primitive_uvs'].numpy()[index_num]
                    intersection_point = (1 - uu - vv) * v0 + uu * v1 + vv * v2
                    intersection_vertices.append(intersection_point)
                    #记录穿透面法线
                    normal = np.cross(v1 - v0, v2 - v0)
                    normal /= np.linalg.norm(normal)
                    normals.append(normal)
                index_num += 1

    return blocking_vertices, intersection_vertices, normals

def local_origin(blocking_points_2d, intersection_points_2d, normals, eps_value = 0.25, min_samples_value = 1):
    # eps_value: 聚类邻域半径，根据点间距调整
    # min_samples_value: 聚类最小邻域点数，根据密度调整
    # 执行DBSCAN聚类
    db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    clusters = db.fit_predict(blocking_points_2d)
    labels = db.labels_
    moved_points = []

    for point, normal in zip(intersection_points_2d, normals):
        direction = normal[[0, 2]]
        direction /= np.linalg.norm(direction)
        plt.arrow(point[0], point[1], direction[0], direction[1],
                  head_width=0.05, head_length=0.1, fc='pink', ec='pink')

    for cluster_id in np.unique(labels):
        # 提取当前簇的点
        cluster_mask = (labels == cluster_id)
        cluster_data = blocking_points_2d[cluster_mask]
        # 计算每个簇所有点的中心点
        cluster_center = np.mean(cluster_data, axis=0)
        plt.scatter(cluster_center[0], cluster_center[1], c='k', marker='x', s=100)
        if len(cluster_data) > 1:
            # print('簇中心点：', cluster_center)
            # print(len(cluster_data))
            # 使用线性回归拟合直线
            reg = LinearRegression().fit(cluster_data[:, 0].reshape(-1, 1), cluster_data[:, 1])
            slope = reg.coef_[0]
            intercept = reg.intercept_

            # 计算预测值
            y_pred = reg.predict(cluster_data[:, 0].reshape(-1, 1))

            # 计算R²和MSE
            r2 = r2_score(cluster_data[:, 1], y_pred)
            mse = mean_squared_error(cluster_data[:, 1], y_pred)

            #print(f'R²: {r2}, MSE: {mse}')
            if r2 < 0.2:
                moved_point = [0, 0]
                moved_points.append(moved_point)
                plt.scatter(moved_point[0], moved_point[1], c='m', marker='o', s=50)
            else:
                # 生成切线的x坐标范围
                x_vals = np.linspace(cluster_data[:, 0].min() - 0.1, cluster_data[:, 0].max() + 0.1, 100)
                y_vals = slope * x_vals + intercept
                plt.plot(x_vals, y_vals, 'k--')
                # 计算垂直于切线的方向
                perpendicular_direction = np.array([-slope, 1])
                perpendicular_direction /= np.linalg.norm(perpendicular_direction)
                # 判断向着切线的哪一侧移动
                directions = []
                for point in cluster_data:
                    idx = np.where(np.isclose(intersection_points_2d, point, atol=1e-5).all(axis=1))[0]
                    if len(idx) > 0:
                        directions.append(normals[idx[0]][[0, 2]])
                avg_direction = np.mean(directions, axis=0)
                avg_direction /= np.linalg.norm(avg_direction)
                if np.dot(perpendicular_direction, avg_direction) > 0:
                    perpendicular_direction = -perpendicular_direction
                # 向局部网格内部方向移动1mm距离
                moved_point = cluster_center + perpendicular_direction * 1.5
                moved_points.append(moved_point)
                plt.scatter(moved_point[0], moved_point[1], c='m', marker='o', s=50)
        else:
            moved_point = [0,0]
            moved_points.append(moved_point)
            plt.scatter(moved_point[0], moved_point[1], c='m', marker='o', s=50)

    return labels, moved_points

def find_shadow_points(mesh, blocking_vertices, blocking_points_2d, labels, local_points, intersection_vertices):
    vertices = np.asarray(mesh.vertices)
    intersection_vertices = np.asarray(intersection_vertices)
    shadow_vertices_all = []
    new_vertices_all = []
    lengths_all = []
    local_points_all = []

    for cluster_id in np.unique(labels):
        # 提取当前簇的点
        cluster_mask = (labels == cluster_id)
        cluster_data = blocking_points_2d[cluster_mask]
        local_origin = local_points[cluster_id]

        for v in cluster_data:
            # 获取对应的真实三维坐标点
            real_point_idx = np.where((blocking_points_2d == v).all(axis=1))[0][0]
            real_point = blocking_vertices[real_point_idx]
            v_xz = v

            # 条件1：距离小于 0.15mm
            xz_projections = vertices[:, [0, 2]]
            distances = np.linalg.norm(xz_projections - v_xz, axis=1)
            mask_condition1 = distances < 0.2

            # 条件2：y坐标在簇发射点和穿透点之间
            intersection_mask = np.all(np.isclose(intersection_vertices[:, [0, 2]], v_xz), axis=1)
            if np.any(intersection_mask):
                min_y_intersection = np.min(intersection_vertices[intersection_mask][:, 1])
                mask_condition2 = (vertices[:, 1] < real_point[1]) & (vertices[:, 1] > min_y_intersection - 0.1)
            else:
                mask_condition2 = np.zeros_like(mask_condition1, dtype=bool)

            # 条件3：到局部原点的距离小于发射点到局部原点的距离
            local_origin_distances = np.linalg.norm(xz_projections - local_origin, axis=1)
            v_xz_norm = np.linalg.norm(v_xz - local_origin) + 0.01
            mask_condition3 = local_origin_distances < v_xz_norm

            # 组合条件
            mask = mask_condition1 & mask_condition2 & mask_condition3
            penetrated_candidates = vertices[mask]

            # 移动阴影点
            for u in penetrated_candidates:
                u_xz = u[[0, 2]]
                u_xz_norms = np.linalg.norm(u_xz - local_origin)
                if u_xz_norms == 0:
                    continue
                length = v_xz_norm - u_xz_norms + 0.005
                new_u_xz = u_xz + (u_xz - local_origin) / u_xz_norms * length
                new_u = np.array([new_u_xz[0], u[1], new_u_xz[1]])
                shadow_vertices_all.append(u)
                new_vertices_all.append(new_u)
                lengths_all.append(length)
                local_points_all.append(local_origin)

    # 去除重复的阴影点
    unique_entries = {}
    for u, new_vertex, length, center in zip(shadow_vertices_all, new_vertices_all, lengths_all, local_points_all):
        key = tuple(np.round(u, 6))
        if key not in unique_entries or length > unique_entries[key]['length']:
            unique_entries[key] = {'u': u, 'new_vertex': new_vertex, 'length': length, 'center': center}

    # 返回阴影点
    shadow_vertices = [entry['u'] for entry in unique_entries.values()]
    new_vertices = [entry['new_vertex'] for entry in unique_entries.values()]
    center_vertices = [entry['center'] for entry in unique_entries.values()]

    return shadow_vertices, new_vertices, center_vertices

def fitune_vertices(new_vertices, center_vertices, distance_threshold=0.6, angle_threshold=10):
    # 精细化调整移动点
    #条件1: 按照y轴值从大到小排序
    sorted_indices = np.argsort(-np.array(new_vertices)[:, 1])
    new_vertices = np.array(new_vertices)[sorted_indices]
    center_vertices = np.array(center_vertices)[sorted_indices]

    for i, current_vertex in enumerate(new_vertices):
        current_xz = current_vertex[[0, 2]]
        for j in range(i + 1, len(new_vertices)):
            other_vertex = new_vertices[j]
            other_xz = other_vertex[[0, 2]]
            distance = np.linalg.norm(current_xz - other_xz)
            # 条件2：该点xz平面投影点与当前点xz平面投影点的距离小于阈值
            if distance < distance_threshold:
                current_center = center_vertices[i][[0, 1]]
                current_distance = np.linalg.norm(current_xz - current_center)
                other_center = center_vertices[j][[0, 1]]
                other_distance = np.linalg.norm(other_xz - other_center)
                # 条件3：该点xz平面投影点到对应中心点的距离小于当前点xz平面投影点到对应中心点的距离
                if other_distance < current_distance:
                    vector1 = other_xz - current_xz
                    vector2 = current_xz - current_center
                    angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
                    #条件4：夹角小于阈值
                    if 180 - angle < angle_threshold:
                        # 计算移动方向和距离
                        move_direction = (other_xz - other_center) / np.linalg.norm(other_xz - other_center)
                        move_distance = current_distance - other_distance  + 0.005
                        new_xz = other_xz + move_direction * move_distance
                        # 更新新坐标位置
                        new_vertices[j][0] = new_xz[0]
                        new_vertices[j][2] = new_xz[1]

    # 还原到未按y轴值排序的顺序
    original_order_indices = np.argsort(sorted_indices)
    new_vertices = new_vertices[original_order_indices]
    return new_vertices

def filling_undercut(mesh, undercut_direction):
    
    # 变换模型使得倒凹方向与 y 轴负方向一致
    trans_mesh, rotation_matrix = align_mesh_to_direction(mesh, undercut_direction)
    #print('Rotation matrix:', rotation_matrix)
    #o3d.io.write_triangle_mesh(path + 'trans_mesh.stl', mesh)
    
    # 寻找发射点与被穿透点
    blocking_vertices, intersection_vertices, normals = find_blocking_vertices(trans_mesh, direction=[0, -1, 0])
    print(f'Number of blocking vertices: {len(blocking_vertices)}')
    
    if len(blocking_vertices) > 0:
    # 聚类寻找簇原点
        blocking_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(blocking_vertices)))
        intersection_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(intersection_vertices)))
        blocking_points_2d = np.asarray(blocking_pc.points)[:, [0, 2]]
        intersection_points_2d = np.asarray(intersection_pc.points)[:, [0, 2]]
        labels, local_points = local_origin(blocking_points_2d, intersection_points_2d, normals)

        # 寻找被遮挡的阴影点，并移动它们
        shadow_vertices, new_vertices, center_vertices = find_shadow_points(trans_mesh, blocking_vertices, blocking_points_2d, labels, local_points, intersection_vertices)
        print(f'Number of shadow vertices: {len(shadow_vertices)}')

        if len(shadow_vertices) > 0:
            #精细化调整移动点
            final_vertices = fitune_vertices(new_vertices, center_vertices)

            # 使用 KDTree 快速查找最近邻索引
            original_vertices = np.asarray(trans_mesh.vertices)
            tree = cKDTree(original_vertices)
            _, point_idx = tree.query(np.array(shadow_vertices))

            # 所有目标坐标（未变形的顶点保持不变，变形的顶点替换为 new_vertices）
            all_point_idx = np.arange(len(original_vertices))
            all_point_dst = original_vertices.copy()
            all_point_dst[point_idx] = np.array(final_vertices)

            # 执行 TPS 变形
            undercut_filled_mesh = tps(
                trimesh.Trimesh(vertices=original_vertices, faces=np.asarray(trans_mesh.triangles)),  # 使用 trimesh.Trimesh 对象
                point_idx=all_point_idx,
                point_dst=all_point_dst,
                lambda_=0.5  # 可调节形变刚度参数
            )
            #undercut_filled_mesh.export(path + 'trans_undercut_filled.stl')

            # 将新模型变换回初始位置
            undercut_filled_mesh_o3d = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(undercut_filled_mesh.vertices),
                triangles=o3d.utility.Vector3iVector(undercut_filled_mesh.faces)
            )

            new_mesh = inverse_transform_mesh(undercut_filled_mesh_o3d, rotation_matrix)
            new_mesh.remove_duplicated_vertices()
            new_mesh.remove_duplicated_triangles()
            new_mesh.remove_non_manifold_edges()
            new_mesh.remove_degenerate_triangles()
            new_mesh.remove_unreferenced_vertices()
            new_mesh.compute_vertex_normals()  # 计算法线

            # 转换为trimesh格式
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(new_mesh.vertices),
                faces=np.asarray(new_mesh.triangles),
                vertex_normals=np.asarray(new_mesh.vertex_normals)
            )

            return tri_mesh

        else:
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_normals=np.asarray(mesh.vertex_normals)
            )
            return tri_mesh

    else:
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals)
        )
        return tri_mesh