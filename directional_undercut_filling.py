import open3d as o3d
import numpy as np
from tps import TPS
import time
import trimesh
from scipy.spatial import cKDTree
# import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import copy

def normalize(v):
    return v / np.linalg.norm(v)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2"""
    a, b = normalize(vec1), normalize(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)  # 平行或反向
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    M = np.eye(4)
    M[:3, :3] = rotation_matrix
    return M, rotation_matrix

def getBoundaryPoints(mesh):
    """get the boundary points of a mesh IN THE ORDER of how they appear in the mesh"""
    if type(mesh) is not trimesh.Trimesh:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
    unique_elements, counts = np.unique(mesh.edges_sorted, return_counts=True, axis=0)
    edges = unique_elements[np.where(counts == 1)[0]]
    boundary_point_id = np.unique(edges.flatten())
    boundary_point = mesh.vertices[boundary_point_id]
    return boundary_point, boundary_point_id

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

def is_point_in_polygon(point, polygon):
    # 使用射线法判断点是否在多边形内
    x, z = point
    n = len(polygon)
    inside = False

    p1x, p1z = polygon[0]
    for i in range(n + 1):
        p2x, p2z = polygon[i % n]
        if z > min(p1z, p2z):
            if z <= max(p1z, p2z):
                if x <= max(p1x, p2x):
                    if p1z != p2z:
                        xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1z = p2x, p2z

    return inside

def margin_and_shadow(mesh, ids):
    # 判断边缘是否处于倒凹阴影内
    # 获取边缘轮廓顶点和其余顶点
    vertices = np.asarray(mesh.vertices)
    edge_vertices = vertices[ids]
    other_vertices = np.delete(vertices, ids, axis=0)

    # 投影到 xz 平面
    edge_vertices_xz = edge_vertices[:, [0, 2]]
    other_vertices_xz = other_vertices[:, [0, 2]]

    # 判断other_vertices_xz中的点是否在多边形外
    outside_points = []
    for point in other_vertices_xz:
        if not is_point_in_polygon(point, edge_vertices_xz):
            outside_points.append(point)
    
    # 将outside_points从other_vertices_xz中移除，得到内部点
    outside_points_set = set(map(tuple, outside_points))
    #inside_points = np.array([point for point in other_vertices_xz if tuple(point) not in outside_points_set])

    return outside_points

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
    #初始化
    patch_num = 0
    patch_points = []
    local_origin_points = []

    #投影法线
    for point, normal in zip(intersection_points_2d, normals):
        direction = normal[[0, 2]]
        direction /= np.linalg.norm(direction)
    
    # eps_value: 聚类邻域半径，根据点间距调整
    # min_samples_value: 聚类最小邻域点数，根据密度调整
    # 执行DBSCAN聚类
    db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    clusters = db.fit_predict(blocking_points_2d)
    labels = db.labels_

    for cluster_id in np.unique(labels):
        # 提取当前簇的点
        cluster_mask = (labels == cluster_id)
        cluster_data = blocking_points_2d[cluster_mask]

        # 计算簇内点之间跨度
        max_x = cluster_data[:, 0].max()
        min_x = cluster_data[:, 0].min()
        max_z = cluster_data[:, 1].max()
        min_z = cluster_data[:, 1].min()
        max_gap = max(max_x - min_x, max_z - min_z)
        if 0 < max_gap <= 1:
            patch = 1
        elif 1 < max_gap <= 2:
            patch = 2
        elif 2 < max_gap <= 3:
            patch = 3
        elif 3 < max_gap <= 4:  
            patch = 4
        elif 4 < max_gap <= 5:
            patch = 5
        elif 5 < max_gap <= 6:
            patch = 6
        elif max_gap > 6:
            patch = 7

        if max_gap > 1:
            # 拆分簇
            kmeans = KMeans(n_clusters=patch, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_data)
            for sub_cluster_id in range(patch):
                sub_cluster_data = cluster_data[sub_labels == sub_cluster_id]
                patch_num += 1
                patch_points.append(sub_cluster_data)

                # 计算每个子簇的中心点
                sub_cluster_center = np.mean(sub_cluster_data, axis=0)
                # 处理子簇的方向和移动
                directions = []
                for point in sub_cluster_data:
                    idx = np.where(np.isclose(intersection_points_2d, point, atol=1e-5).all(axis=1))[0]
                    if len(idx) > 0:
                        directions.append(normals[idx[0]][[0, 2]])
                avg_direction = np.mean(directions, axis=0)
                avg_direction /= np.linalg.norm(avg_direction)
                # 沿法线平均方向的反方向移动1.5mm
                moved_point = sub_cluster_center + (-avg_direction) * 1.5
                local_origin_points.append(moved_point)
            
        else:
            patch_num += 1
            patch_points.append(cluster_data)
            # 计算每个簇所有点的中心点
            cluster_center = np.mean(cluster_data, axis=0)

            directions = []
            for point in cluster_data:
                idx = np.where(np.isclose(intersection_points_2d, point, atol=1e-5).all(axis=1))[0]
                if len(idx) > 0:
                    directions.append(normals[idx[0]][[0, 2]])
            avg_direction = np.mean(directions, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)
            # 沿法线平均方向的反方向移动1.5mm
            moved_point = cluster_center + (-avg_direction) * 1.5
            local_origin_points.append(moved_point)

    return patch_num, patch_points, local_origin_points

def find_shadow_points(mesh, blocking_vertices, blocking_points_2d, patch_num, patch_points, local_origin_points, intersection_vertices):
    vertices = np.asarray(mesh.vertices)
    intersection_vertices = np.asarray(intersection_vertices)
    shadow_vertices_all = []
    new_vertices_all = []
    lengths_all = []
    local_points_all = []

    for cluster_id in range(patch_num):
        #print('cluster_id:', cluster_id)
        # 提取当前簇的点
        cluster_data = patch_points[cluster_id]
        local_origin = local_origin_points[cluster_id]
        # print('cluster_data:', cluster_data)
        # print('local_origin:', local_origin)

        for v in cluster_data:
            # 获取对应的真实三维坐标点
            real_point_idx = np.where((blocking_points_2d == v).all(axis=1))[0][0]
            real_point = blocking_vertices[real_point_idx]
            v_xz = v

            # 条件1：距离小于 0.3mm
            xz_projections = vertices[:, [0, 2]]
            distances = np.linalg.norm(xz_projections - v_xz, axis=1)
            mask_condition1 = distances < 0.3

            # 条件2：y坐标在簇发射点和穿透点之间
            intersection_mask = np.all(np.isclose(intersection_vertices[:, [0, 2]], v_xz), axis=1)
            if np.any(intersection_mask):
                min_y_intersection = np.min(intersection_vertices[intersection_mask][:, 1])
                mask_condition2 = (vertices[:, 1] < real_point[1] - 0.05) & (vertices[:, 1] > min_y_intersection - 0.1)
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

def fitune_vertices(new_vertices, center_vertices, distance_threshold=0.6, angle_threshold=25):
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

def filling_undercut(mesh, undercut_direction, save_path=None, display=False):

    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    # 新增网格简化逻辑
    target_triangles = 8000
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    
    # 变换模型使得倒凹方向与 y 轴负方向一致
    # trans_mesh, rotation_matrix = align_mesh_to_direction(mesh, undercut_direction)
    M, rotation_matrix = rotation_matrix_from_vectors(undercut_direction, [0, -1, 0])
    trans_mesh = copy.copy(mesh)
    vertices = np.asarray(trans_mesh.vertices)
    transformed_vertices = vertices @ rotation_matrix.T
    trans_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    print('Rotation matrix:', rotation_matrix)
    
    #o3d.io.write_triangle_mesh(path + 'trans_mesh.stl', mesh)

    # 判断边缘是否处于倒凹阴影内，ids为带顺序的边缘顶点索引
    # outside_points = margin_and_shadow(trans_mesh, ids)
    # print(f'Number of outside points: {len(outside_points)}')
    
    # 寻找发射点与被穿透点
    blocking_vertices, intersection_vertices, normals = find_blocking_vertices(trans_mesh, direction=[0, -1, 0])
    print(f'Number of blocking vertices: {len(blocking_vertices)}')
    print(f'Number of intersection vertices: {len(intersection_vertices)}')

    if display:
        vis_list1 = [trans_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)] # 可视化列表
        blocking_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(blocking_vertices)))
        blocking_pc.paint_uniform_color([1, 0, 0])
        intersection_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(intersection_vertices)))
        intersection_pc.paint_uniform_color([1, 1, 0])
        vis_list1 += [blocking_pc, intersection_pc]
        o3d.visualization.draw_geometries(vis_list1)
    
    if len(blocking_vertices) > 0:
    # 聚类寻找簇原点
        blocking_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(blocking_vertices)))
        intersection_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(intersection_vertices)))
        blocking_points_2d = np.asarray(blocking_pc.points)[:, [0, 2]]
        intersection_points_2d = np.asarray(intersection_pc.points)[:, [0, 2]]
        patch_num, patch_points, local_origin_points = local_origin(blocking_points_2d, intersection_points_2d, normals)

        # 寻找被遮挡的阴影点，并移动它们
        shadow_vertices, new_vertices, center_vertices = find_shadow_points(trans_mesh, blocking_vertices, blocking_points_2d, patch_num, patch_points, local_origin_points, intersection_vertices)
        print(f'Number of shadow vertices: {len(shadow_vertices)}')
        print(f'Number of new vertices: {len(new_vertices)}')
        print(f'Number of center vertices: {len(center_vertices)}')


        if len(shadow_vertices) > 0:
            #精细化调整移动点
            final_vertices = fitune_vertices(new_vertices, center_vertices)

            if display:
                vis_list2 = [trans_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)] # 可视化列表
                blocking_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(blocking_vertices)))
                blocking_pc.paint_uniform_color([1, 0, 0])
                intersection_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(intersection_vertices)))
                intersection_pc.paint_uniform_color([1, 1, 0])
                shadow_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(shadow_vertices)))
                shadow_pc.paint_uniform_color([0, 0, 1])
                new_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(new_vertices)))
                final_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(final_vertices)))
                final_pc.paint_uniform_color([0, 1, 0])

                # x-z平面
                shadow_points_2d = np.asarray(shadow_pc.points)[:, [0, 2]]
                new_points_2d = np.asarray(new_pc.points)[:, [0, 2]]
                final_points_2d = np.asarray(final_pc.points)[:, [0, 2]]

                # 创建二维图像A
                plt.figure(figsize=(10, 10))
                plt.scatter(shadow_points_2d[:, 0], shadow_points_2d[:, 1],
                            c='b', alpha=0.3, label='Shadow Vertices')
                plt.scatter(blocking_points_2d[:, 0], blocking_points_2d[:, 1],
                            c='r', alpha=0.8, label='Blocking Vertices')
                plt.scatter(intersection_points_2d[:, 0], intersection_points_2d[:, 1],
                            c='y', alpha=0.5, label='Intersection Vertices')
                plt.scatter(new_points_2d[:, 0], new_points_2d[:, 1],
                            c='g', alpha=0.5, label='New Vertices')

                plt.gca().invert_yaxis()  # 反转Z轴方向
                # 设置图像属性
                plt.title(f'2D Projection of Vertices (Y-Axis Discarded)')
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # 保存和显示
                # plt.savefig(path + 'all_verticesA.png', dpi=300)
                plt.show()

                # 创建二维图像B
                plt.figure(figsize=(10, 10))
                plt.scatter(new_points_2d[:, 0], new_points_2d[:, 1],
                            c='m', alpha=0.5, label='New Vertices')
                plt.scatter(final_points_2d[:, 0], final_points_2d[:, 1],
                            c='g', alpha=0.5, label='Final Vertices')

                plt.gca().invert_yaxis()  # 反转Z轴方向
                # 设置图像属性
                plt.title(f'2D Projection of Vertices (Y-Axis Discarded)')
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # 保存和显示
                # plt.savefig(path + 'all_verticesB.png', dpi=300)
                plt.show()

                vis_list2 += [intersection_pc, blocking_pc, shadow_pc, final_pc]
                o3d.visualization.draw_geometries(vis_list2)
            
            # 使用 KDTree 快速查找最近邻索引
            original_vertices = np.asarray(trans_mesh.vertices)
            tree = cKDTree(original_vertices)
            _, point_idx = tree.query(np.array(shadow_vertices))

            bound_points, bound_indices = getBoundaryPoints(trans_mesh)
            dist_boundary = cdist(original_vertices[point_idx], bound_points).min(axis=1)
            dist_mask = dist_boundary > 0.2
            # boundary_num = np.sum(~dist_mask)

            # 所有目标坐标（未变形的顶点保持不变，变形的顶点替换为 new_vertices）
            all_point_idx = np.arange(len(original_vertices))
            all_point_dst = original_vertices.copy()
            boundary_dist = np.linalg.norm(all_point_dst[point_idx[~dist_mask]] - np.array(final_vertices)[~dist_mask], axis=1)
            boundary_num = np.sum(boundary_dist > 0.03)
            all_dist = np.linalg.norm(all_point_dst[point_idx] - np.array(final_vertices), axis=1)
            bound_ori = all_point_dst[point_idx[~dist_mask]]
            bound_new = np.array(final_vertices)[~dist_mask]
            tps_ori = all_point_dst[point_idx[dist_mask]]
            tps_new = np.array(final_vertices)[dist_mask]
            dmin, dmax = 0.0, 1.0
            t = np.clip((all_dist - dmin) / (dmax - dmin), 0, 1)
            # viridis = np.array([
            #     [68, 1, 84],
            #     [59, 82, 139],
            #     [33, 144, 140],
            #     [94, 201, 98],
            #     [253, 231, 37]
            # ]) 
            viridis = np.array([[255,0,0], [255,255,255], [0,0,255]])
            rgb = np.clip(
                np.round([
                    np.interp(t, np.linspace(0, 1, viridis.shape[0]), viridis[:, 0]),
                    np.interp(t, np.linspace(0, 1, viridis.shape[0]), viridis[:, 1]),
                    np.interp(t, np.linspace(0, 1, viridis.shape[0]), viridis[:, 2])
                ]).T.astype(np.uint8),
                0, 255
            )
            rgb_bound = rgb[~dist_mask]
            rgb_tps = rgb[dist_mask]
            out_info = {
                'boundary_num': boundary_num,
                'bound_ori': bound_ori,
                'bound_new': bound_new,
                'rgb_bound': rgb_bound,
                'tps_ori': tps_ori,
                'tps_new': tps_new,
                'rgb_tps': rgb_tps
            }

            all_point_dst[point_idx[dist_mask]] = np.array(final_vertices)[dist_mask]

            # 执行 TPS 变形
            # undercut_filled_mesh = tps(
            #     trimesh.Trimesh(vertices=original_vertices, faces=np.asarray(trans_mesh.triangles)),  # 使用 trimesh.Trimesh 对象
            #     point_idx=all_point_idx,
            #     point_dst=all_point_dst,
            #     lambda_=0.1  # 可调节形变刚度参数
            # )
            trans = TPS(original_vertices[all_point_idx], all_point_dst, lambda_=0.5)
            undercut_filled_mesh = trimesh.Trimesh(trans(np.asarray(trans_mesh.vertices)), trans_mesh.triangles)

            # 将新模型变换回初始位置
            undercut_filled_mesh_o3d = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(undercut_filled_mesh.vertices),
                triangles=o3d.utility.Vector3iVector(undercut_filled_mesh.faces)
            )
            if save_path:
                undercut_filled_mesh_o3d.compute_vertex_normals()
                o3d.io.write_triangle_mesh(f'{save_path}/undercut.stl', undercut_filled_mesh_o3d)

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

            out_info['rotation_matrix'] = rotation_matrix

            return tri_mesh, out_info

        else:
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_normals=np.asarray(mesh.vertex_normals)
            )
            return tri_mesh, None

    else:
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals)
        )
        return tri_mesh, None
