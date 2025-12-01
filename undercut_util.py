import time
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import trimesh
from scipy import stats

def read_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """Return a o3d.geometry.TriangleMesh object from a file path"""
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

def getBoundaryPoints(mesh):
    """get the boundary points of a mesh IN THE ORDER of how they appear in the mesh"""
    if type(mesh) is not trimesh.Trimesh:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
    unique_elements, counts = np.unique(mesh.edges_sorted, return_counts=True, axis=0)
    edges = unique_elements[np.where(counts == 1)[0]]
    boundary_point_id = np.unique(edges.flatten())
    boundary_point = mesh.vertices[boundary_point_id]
    return boundary_point, boundary_point_id


def robust_outlier_detection(data, target_index=0, alpha=0.05):
    """稳健的离群点检测"""
    
    # 1. F检验
    var_full = np.var(data, ddof=1)
    data_without = np.delete(data, target_index)
    var_without = np.var(data_without, ddof=1)
    
    F_stat = max(var_full/var_without, var_without/var_full)
    df1, df2 = len(data)-1, len(data_without)-1
    p_value = 2 * min(stats.f.cdf(F_stat, df1, df2), 
                      1 - stats.f.cdf(F_stat, df1, df2))
    
    # 2. 方差变化比例
    variance_change_ratio = (var_full - var_without) / var_full * 100
    
    # 3. Z-score
    z_score = abs(data[target_index] - np.mean(data)) / np.std(data, ddof=1)
    
    print("=== 综合离群点检测 ===")
    print(f"F检验 p值: {p_value:.6f}")
    print(f"方差变化: {variance_change_ratio:.2f}%")
    print(f"Z-score: {z_score:.3f}")
    
    # 综合判断
    if p_value < alpha and variance_change_ratio > 5 and z_score > 2:
        print("🔴 结论: 很可能是离群点")
        return True
    elif (p_value < alpha) or (variance_change_ratio > 10) or (z_score > 2.5):
        print("🟡 结论: 可能是离群点")
        return True
    else:
        print("🟢 结论: 不太可能是离群点")
        return False

def angle_with_y_axis(vectors):
    """
    计算每个向量与 [0, 1, 0] (Y轴正方向) 的夹角
    
    参数:
    vectors: numpy数组, 形状为 (n, 3) 的向量集合
    
    返回:
    angles: numpy数组, 形状为 (n,) 的夹角(角度制)
    """
    # 参考向量 [0, 1, 0]
    reference = np.array([0, 1, 0])
    
    # 计算点积: 由于参考向量是[0,1,0]，点积就是每个向量的y分量
    dot_products = vectors[:, 1]  # 直接取y分量
    
    # 计算向量的模长
    norms = np.linalg.norm(vectors, axis=1)
    
    # 计算夹角余弦值，确保在[-1, 1]范围内
    cos_angles = dot_products / norms
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # 计算夹角(弧度)并转换为角度
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg

def get_insert_direction(prep_mesh: o3d.geometry.TriangleMesh, vis=False) -> np.ndarray:
    """Visulize the computed minimum-undercut direction of a prep tooth"""
    # 加载网格文件
    s0 = time.time()
    # 连接准备网格和邻接网格
    prep_mesh.compute_vertex_normals()
    prep_mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(prep_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(prep_mesh_o3d_t)  # 将网格添加到光线投射场景中

    # 球体的上半部分的所有点，用来和原点连接产生光线方向向量
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=50)
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
    s1 = time.time()
    print(f"sample rays: {s1 - s0}")

    out_undercut_num = []
    hit_num = []
    # 遍历每个点进行光线交集测试
    print(len(points))
    for i, p in enumerate(points):
        if i == 0:
            print(1)
        ray_directions = np.array([p]).repeat(
            len(ray_origins), axis=0
        )  # 为所有起点重复方向向量
        if i == 0:
            print(2)
        rays = np.concatenate(
            (ray_origins, ray_directions * -1), axis=1
        )  # 组合起点和方向
        if i == 0:
            print(3)
        ray_intersections = scene.count_intersections(
            o3d.core.Tensor(rays.astype(np.float32))
        )  # 测试遮挡
        if i == 0:
            print(4)
        undercut_num = len(np.where(ray_intersections.numpy() > 1)[0])
        if i == 0:
            print(5)
        hit_num.append(len(np.where(ray_intersections.numpy() > 0)[0]))
        if i == 0:
            print(6)
        out_undercut_num.append(undercut_num)
        if i == 0:
            print(7)
        # ################################################################################################
        # if vis:
        #     rays_o3d = o3d.geometry.LineSet()
        #     line_pts = []
        #     for ray in rays:
        #         line_pts.append([ray[0], ray[1], ray[2]])
        #         line_pts.append([ray[0]+ray[3]*10, ray[1]+ray[4]*10, ray[2]+ray[5]*10])
        #     rays_o3d.points = o3d.utility.Vector3dVector(line_pts)
        #     rays_o3d.lines = o3d.utility.Vector2iVector([[2*i, 2*i+1] for i in range(rays.shape[0])])
        #     pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))
        #     mesh_o3d = mesh_concat
        #     mesh_o3d.compute_vertex_normals()
        #     o3d.visualization.draw_geometries([rays_o3d, pc, mesh_o3d])
        # ################################################################################################
    # 倒凹最小的前1/4个方向的平均值
    s2 = time.time()
    print(f"loop time: {s2 - s1}")
    min_indices_num = np.argsort(out_undercut_num)[: int(points.shape[0] / 4)]
    out_directions = points[min_indices_num]
    out_direction = np.mean(points[min_indices_num], axis=0)[np.newaxis, ...]

    # def angle_between(v1, v2):
    #     v1 /= np.linalg.norm(v1)
    #     v2 /= np.linalg.norm(v2)
    #     return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # mean_norm = np.mean(np.asarray(prep_mesh.vertex_normals), axis=0)
    # if angle_between(mean_norm, out_direction[0]) < np.pi / 2:
    #     out_direction[0] *= -1.0
    if out_direction[0][1] > 0:
        out_direction[0] *= -1.0
    return out_direction[0]
    print(f"find min-undercut ray: {time.time() - s0}")
    print(out_direction)
    # undercut blockout

    ################################################################################################
    # vis
    if vis:
        proposed_directions_o3d = o3d.geometry.LineSet()
        line_pts = []
        for p in out_directions:
            line_pts.extend([[0, 0, 0], [p[0], p[1], p[2]]])
        proposed_directions_o3d.points = o3d.utility.Vector3dVector(
            np.array(line_pts) * 3
        )
        proposed_directions_o3d.lines = o3d.utility.Vector2iVector(
            [[i * 2, i * 2 + 1] for i in range(len(out_directions))]
        )
        proposed_directions_o3d.colors = o3d.utility.Vector3dVector(
            [[0, 0, 1] for i in range(len(out_directions))]
        )
        final_direction_o3d = o3d.geometry.LineSet()
        final_direction_o3d.points = o3d.utility.Vector3dVector(
            np.array([np.array([0, 0, 0]), out_direction[0] * 10])
        )
        final_direction_o3d.lines = o3d.utility.Vector2iVector([[0, 1]])
        final_direction_o3d.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([final_direction_o3d, frame, prep_mesh, pc])
    ################################################################################################


def get_insert_direction_copy(
    prep_mesh: o3d.geometry.TriangleMesh, vis=False
) -> np.ndarray:
    """Visulize the computed minimum-undercut direction of a prep tooth"""
    bound_points, _ = getBoundaryPoints(prep_mesh)
    max_dist_boundary = max(cdist(np.asarray(prep_mesh.vertices), bound_points).min(axis=1))
    # 加载网格文件
    s0 = time.time()
    # 连接准备网格和邻接网格
    prep_mesh.compute_vertex_normals()
    prep_mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(prep_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(prep_mesh_o3d_t)  # 将网格添加到光线投射场景中

    # 球体的上半部分的所有点，用来和原点连接产生光线方向向量
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=50)
    points = np.asarray(sphere.vertices)
    points = points[points[:, 1] > 0.8]
    # 找到 prep_mesh 的 y 值最高点
    max_y_prep_mesh = np.asarray(prep_mesh.vertices)[:, 1].max()
    # 生成均匀的点阵作为光线起点
    grid_size = 150  # 点阵大小，可以根据需要调整
    x_range = np.linspace(
        prep_mesh.get_max_bound()[0]  + (prep_mesh.get_max_bound()[0] - prep_mesh.get_center()[0]) * 2, 
        prep_mesh.get_min_bound()[0]  - (prep_mesh.get_center()[0] - prep_mesh.get_min_bound()[0]) * 2, grid_size
    )
    z_range = np.linspace(
        prep_mesh.get_max_bound()[2]  + (prep_mesh.get_max_bound()[2] - prep_mesh.get_center()[2]) * 2, 
        prep_mesh.get_min_bound()[2]  - (prep_mesh.get_center()[2] - prep_mesh.get_min_bound()[2]) * 2, grid_size
    )
    x_grid, z_grid = np.meshgrid(x_range, z_range)
    y_grid = np.full(
        x_grid.shape, max_y_prep_mesh + 1
    )  # y 坐标高于 prep_mesh 的 y 值最高点
    ray_origins = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    s1 = time.time()
    print(f"sample rays: {s1 - s0}")

    out_undercut_num = []
    hit_num = []
    # 遍历每个点进行光线交集测试
    print(len(points))
    for i, p in enumerate(points):
        # if i == 0:
        #     print(1)
        ray_directions = np.array([p]).repeat(
            len(ray_origins), axis=0
        )  # 为所有起点重复方向向量
        # if i == 0:
        #     print(2)
        rays = np.concatenate(
            (ray_origins, ray_directions * -1), axis=1
        )  # 组合起点和方向
        # if i == 0:
        #     print(3)
        ray_intersections = scene.count_intersections(
            o3d.core.Tensor(rays.astype(np.float32))
        )  # 测试遮挡
        lx = scene.list_intersections(
            o3d.core.Tensor(rays.astype(np.float32))
        )  # 获取所有交点
        lx = {k:v.numpy() for k,v in lx.items()}
        v = np.asarray(prep_mesh.vertices)
        t = np.asarray(prep_mesh.triangles)
        tidx = lx['primitive_ids']
        uv = lx['primitive_uvs']
        w = 1 - np.sum(uv, axis=1)
        c = \
        v[t[tidx, 1].flatten(), :] * uv[:, 0][:, None] + \
        v[t[tidx, 2].flatten(), :] * uv[:, 1][:, None] + \
        v[t[tidx, 0].flatten(), :] * w[:, None]
        ray_splits = lx['ray_splits']
        intersection_points = []
        for ray_id, (start, end) in enumerate(zip(ray_splits[:-1], ray_splits[1:])):
            intersection_info = c[start:end]
            if len(intersection_info) > 1:
                for i,t in enumerate(intersection_info):
                    intersection_points.append(t)
        if len(intersection_points):
            dist_boundary = cdist(np.array(intersection_points), bound_points).min(axis=1)
            dist_weight = max_dist_boundary + 1 - dist_boundary
            undercut_num = np.sum(dist_weight)
            out_undercut_num.append(undercut_num)
        else:
            out_undercut_num.append(0)
        # if i == 0:
        #     print(4)
        # undercut_num = len(np.where(ray_intersections.numpy() > 1)[0])
        # if i == 0:
        #     print(5)
        # hit_num.append(len(np.where(ray_intersections.numpy() > 0)[0]))
        # if i == 0:
        #     print(6)
        # out_undercut_num.append(undercut_num)
        # if i == 0:
        #     print(7)
        # ################################################################################################
        # if vis:
        #     rays_o3d = o3d.geometry.LineSet()
        #     line_pts = []
        #     for ray in rays:
        #         line_pts.append([ray[0], ray[1], ray[2]])
        #         line_pts.append([ray[0]+ray[3]*10, ray[1]+ray[4]*10, ray[2]+ray[5]*10])
        #     rays_o3d.points = o3d.utility.Vector3dVector(line_pts)
        #     rays_o3d.lines = o3d.utility.Vector2iVector([[2*i, 2*i+1] for i in range(rays.shape[0])])
        #     pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))
        #     mesh_o3d = mesh_concat
        #     mesh_o3d.compute_vertex_normals()
        #     o3d.visualization.draw_geometries([rays_o3d, pc, mesh_o3d])
        # ################################################################################################
    # 倒凹最小的前1/4个方向的平均值
    s2 = time.time()
    print(f"loop time: {s2 - s1}")
    min_indices_num = np.argsort(out_undercut_num)[: int(points.shape[0] / 4)]
    out_directions = points[min_indices_num]
    angles = angle_with_y_axis(out_directions)
    for i in range(len(angles)):
        if robust_outlier_detection(angles, i):
            print(f"remove outlier direction: {out_directions[i]}")
            out_directions[i] = out_directions.mean(axis=0)
        else:
            out_direction = out_directions[i][np.newaxis, ...]
            break

    # out_direction_ = np.mean(points[min_indices_num], axis=0)[np.newaxis, ...]
    # angle = abs(angle_with_y_axis(out_direction_) - angle_with_y_axis(out_direction))
    # print(angle)
    # out_direction = out_directions[0][np.newaxis, ...]
    # out_direction = points[np.argsort(out_undercut_num)[2]]

    if out_direction[0][1] > 0:
        out_direction[0] *= -1.0
    return out_direction[0]
    # return out_direction[0], angle
    # if out_direction[1] > 0:
    #     out_direction *= -1.0
    # return out_direction
    print(f"find min-undercut ray: {time.time() - s0}")
    print(out_direction)
    # undercut blockout

    ################################################################################################
    # vis
    if vis:
        proposed_directions_o3d = o3d.geometry.LineSet()
        line_pts = []
        for p in out_directions:
            line_pts.extend([[0, 0, 0], [p[0], p[1], p[2]]])
        proposed_directions_o3d.points = o3d.utility.Vector3dVector(
            np.array(line_pts) * 3
        )
        proposed_directions_o3d.lines = o3d.utility.Vector2iVector(
            [[i * 2, i * 2 + 1] for i in range(len(out_directions))]
        )
        proposed_directions_o3d.colors = o3d.utility.Vector3dVector(
            [[0, 0, 1] for i in range(len(out_directions))]
        )
        final_direction_o3d = o3d.geometry.LineSet()
        final_direction_o3d.points = o3d.utility.Vector3dVector(
            np.array([np.array([0, 0, 0]), out_direction[0] * 10])
        )
        final_direction_o3d.lines = o3d.utility.Vector2iVector([[0, 1]])
        final_direction_o3d.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([final_direction_o3d, frame, prep_mesh, pc])
    ################################################################################################


if __name__ == "__main__":
    prep_mesh = read_mesh("./b.stl")
    print(get_insert_direction(prep_mesh, vis=False))
