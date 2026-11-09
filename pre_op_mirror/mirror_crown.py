import base64
import copy
import random

import DracoPy
import numpy as np
import open3d as o3d
import py_prepMorphing as morph
import trimesh
from sklearn.neighbors import NearestNeighbors

from dental_arch_curve import DAC


def bio_morphing(bio_crown, std_crown, dstOrient=None, do_reg=True):
    V_template = std_crown.vertices.astype(np.float64)
    F_template = std_crown.faces.astype(np.int32)
    V_to = bio_crown.vertices.astype(np.float64)
    F_to = bio_crown.faces.astype(np.int32)

    if do_reg:
        if dstOrient is None:
            dstOrient = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        success, V_reg, F_reg = morph.GeneralRegistration(
            V_template, F_template, V_to, F_to, 50, dstOrient, True
        )
        print("GeneralRegistration success:", success)
    else:
        V_reg, F_reg = V_template, F_template
    success, V_rebuild, F_rebuild = morph.ItersMorphingAndRebuild(
        V_reg, F_reg, V_to, F_to, isRemoveBoundary=False, removeBoundaryLayerNum=10
    )
    # ItersMorphingAndRebuild 参数说明
    # -------------------------------------------------
    # V_from: np.ndarray (N, 3)   # 源网格顶点坐标
    # F_from: np.ndarray (M, 3)   # 源网格三角面索引
    # V_to:   np.ndarray (N, 3)   # 目标网格顶点坐标
    # F_to:   np.ndarray (M, 3)   # 目标网格三角面索引
    #
    # insertStep: float = 0.3     # 加密步长；越小 → 网格越密、精度越高
    # tpsIters:   int   = 50      # TPS 变形迭代次数
    #
    # isOutAdjustmentIters: bool = True   # 是否循环执行 TPS + 调整逻辑
    # isSplicing:           bool = True   # 是否执行拼接
    # isRemoveBoundary:     bool = False  # 是否移除目标网格边界
    # removeBoundaryLayerNum: int = 3     # 移除的边界层数（isRemoveBoundary=True 时生效）
    #
    # 返回值
    # success: int            # >0 表示算法成功
    # V_out:   np.ndarray     # 变形后的网格顶点 (N, 3)
    # F_out:   np.ndarray     # 变形后的三角面索引 (M, 3)
    return trimesh.Trimesh(V_rebuild, F_rebuild), trimesh.Trimesh(V_reg, F_reg)


def get_mirror_plane(a, b):
    n = np.mean(a - b, axis=0)
    n = n / np.linalg.norm(n)
    center = np.mean((a + b) / 2, axis=0)
    d = np.dot(n, center)
    return n, d


def mirror_mesh(V, F, n, d):
    """
    V : (N,3) 顶点
    F : (M,3) 面索引
    n : (3,)  单位法向量
    d : float 平面方程 n·x+d=0 里的 d
    返回 (V_mirrored, F_mirrored)
    """
    # 1. 顶点反射
    dist = (V @ n + d)[:, None]  # (N,1)
    if dist[0] < 0:
        dist = -(V @ -n + d)[:, None]
    V_mir = V - 2 * dist * n  # (N,3)

    # 2. 翻转面朝向（可选，但强烈建议）
    if F is not None:
        F_mir = F[:, ::-1]  # 把每个三角形的顶点顺序反转
    else:
        F_mir = None
    return V_mir, F_mir


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F)


def adjust_crown_position(crown, other_teeth, beiya_tid=None):
    # 实现牙冠与其他牙齿的位置关系调整的逻辑
    crown_copy = crown.copy()
    crown_copy = crown_copy.simplify_quadratic_decimation(10000)
    adjacent_teeth = []
    beiya_teeth = []
    c = trimesh.collision.CollisionManager()
    c_crown = trimesh.collision.CollisionManager()
    c_crown.add_object("crown", crown)
    for i, tooth in enumerate(other_teeth):
        dist = trimesh.base.proximity.closest_point(
            crown_copy, random.sample(tooth.vertices.tolist(), 100)
        )[1]
        if min(dist) < 1:  # 如果距离小于0.5mm，认为是邻牙
            adjacent_teeth.append(tooth)
            if beiya_tid[i] == 1:
                beiya_teeth.append(1)
            else:
                beiya_teeth.append(0)
    if len(adjacent_teeth) == 0:
        return crown  # 没有邻牙，直接返回
    elif len(adjacent_teeth) == 1:
        if adjacent_teeth[0].bounds[0][0] > crown.bounds[0][0]:
            if beiya_teeth[0] == 1:
                c.add_object("mesh1", adjacent_teeth[0].bounding_box)
            else:
                c.add_object("mesh1", adjacent_teeth[0])
        else:
            if beiya_teeth[0] == 1:
                c.add_object("mesh2", adjacent_teeth[0].bounding_box)
            else:
                c.add_object("mesh2", adjacent_teeth[0])
        is_collision = c.in_collision_single(
            crown,
            return_names=True,
        )
        if is_collision[0]:
            # 有碰撞，对牙冠进行缩小处理
            for i in range(100, 0, -5):
                crown_copy_2 = crown_copy.copy()
                crown_copy_2.apply_transform(
                    trimesh.transformations.compose_matrix([i / 100, 1, 1])
                )
                is_collision = c.in_collision_single(
                    crown_copy_2,
                    return_names=True,
                )
                if not is_collision[0]:
                    crown.apply_transform(
                        trimesh.transformations.compose_matrix([(i + 5) / 100, 1, 1])
                    )
                    return crown
        else:
            # 无碰撞，对牙冠进行放大处理
            for i in range(100, 200, 5):
                crown_copy_2 = crown_copy.copy()
                crown_copy_2.apply_transform(
                    trimesh.transformations.compose_matrix([i / 100, 1, 1])
                )
                is_collision = c.in_collision_single(
                    crown_copy_2,
                    transform=trimesh.transformations.compose_matrix([i / 100, 1, 1]),
                    return_names=True,
                )
                if is_collision[0]:
                    crown.apply_transform(
                        trimesh.transformations.compose_matrix([i / 100, 1, 1])
                    )
                    return crown
    elif len(adjacent_teeth) == 2:
        if adjacent_teeth[0].bounds[0][0] > adjacent_teeth[1].bounds[0][0]:
            if beiya_teeth[0] == 1:
                c.add_object("mesh1", adjacent_teeth[0].bounding_box)
            else:
                c.add_object("mesh1", adjacent_teeth[0])
            if beiya_teeth[1] == 1:
                c.add_object("mesh2", adjacent_teeth[1].bounding_box)
            else:
                c.add_object("mesh2", adjacent_teeth[1])
        else:
            if beiya_teeth[1] == 1:
                c.add_object("mesh1", adjacent_teeth[1].bounding_box)
            else:
                c.add_object("mesh1", adjacent_teeth[1])
            if beiya_teeth[0] == 1:
                c.add_object("mesh2", adjacent_teeth[0].bounding_box)
            else:
                c.add_object("mesh2", adjacent_teeth[0])
        is_collision = c.in_collision_single(
            crown,
            return_names=True,
        )
        if is_collision[0]:
            if "mesh1" in list(is_collision[1]) and "mesh2" in list(is_collision[1]):
                # 同时与两个邻牙碰撞，对牙冠进行缩小处理
                shift = 0
                for i in range(100, 0, -2):
                    crown_copy_2 = crown_copy.copy()
                    crown_copy_2.apply_transform(
                        trimesh.transformations.compose_matrix(
                            scale=[i / 100, 1, 1],
                            translate=[shift / 100, 0, 0],
                        )
                    )
                    is_collision = c.in_collision_single(
                        crown_copy_2,
                        return_names=True,
                    )
                    if not is_collision[0]:
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[(i + 2) / 100, 1, 1],
                                translate=[shift / 100, 0, 0],
                            )
                        )
                        return crown
                    elif list(is_collision[1]) == ["mesh1"]:
                        for j in range(shift, -200, -5):
                            crown_copy_2 = crown_copy.copy()
                            crown_copy_2.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[i / 100, 1, 1],
                                    translate=[j / 100, 0, 0],
                                )
                            )

                            is_collision = c.in_collision_single(
                                crown_copy_2,
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh2"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[i / 100, 1, 1],
                                        translate=[(j + 5) / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                                is_collision[1]
                            ):
                                shift = j
                                break
                            elif not is_collision[0]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(i + 2) / 100, 1, 1],
                                        translate=[j / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif list(is_collision[1]) == ["mesh2"]:
                        for j in range(shift, 200, 5):
                            crown_copy_2 = crown_copy.copy()
                            crown_copy_2.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[i / 100, 1, 1],
                                    translate=[j / 100, 0, 0],
                                )
                            )

                            is_collision = c.in_collision_single(
                                crown_copy_2,
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh1"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[i / 100, 1, 1],
                                        translate=[(j - 5) / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                                is_collision[1]
                            ):
                                shift = j
                                break
                            elif not is_collision:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(i + 2) / 100, 1, 1],
                                        translate=[j / 100, 0, 0],
                                    )
                                )
                                return crown

            elif list(is_collision[1]) == ["mesh1"]:
                # 只与近中邻牙碰撞，对牙冠先平移后放缩
                scale = 100
                for i in range(0, -100, -2):
                    crown_copy_2 = crown_copy.copy()
                    crown_copy_2.apply_transform(
                        trimesh.transformations.compose_matrix(
                            scale=[scale / 100, 1, 1],
                            translate=[i / 100, 0, 0],
                        )
                    )

                    is_collision = c.in_collision_single(
                        crown_copy_2,
                        return_names=True,
                    )
                    if not is_collision[0]:
                        scale += 2
                        crown_copy_2 = crown_copy.copy()
                        crown_copy_2.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                            )
                        )
                        is_collision = c.in_collision_single(
                            crown_copy_2,
                            return_names=True,
                        )
                        if "mesh1" in list(is_collision[1]) and "mesh2" in list(
                            is_collision[1]
                        ):
                            crown.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[scale / 100, 1, 1],
                                    translate=[i / 100, 0, 0],
                                )
                            )
                            return crown
                    elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                        is_collision[1]
                    ):
                        for j in range(scale, 0, -2):
                            crown_copy_2 = crown_copy.copy()
                            crown_copy_2.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[j / 100, 1, 1], translate=[i / 100, 0, 0]
                                )
                            )
                            is_collision = c.in_collision_single(
                                crown_copy_2,
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh1"]:
                                scale = j
                                break
                            elif list(is_collision[1]) == ["mesh2"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 2) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif not is_collision[0]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 2) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif list(is_collision[1]) == ["mesh2"]:
                        scale += 2
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1],
                                translate=[(i + 2.5) / 100, 0, 0],
                            )
                        )
                        return crown

            elif list(is_collision[1]) == ["mesh2"]:
                # 只与远中邻牙碰撞，对牙冠先平移后放缩
                scale = 100
                for i in range(0, 100, 5):
                    crown_copy_2 = crown_copy.copy()
                    crown_copy_2.apply_transform(
                        trimesh.transformations.compose_matrix(
                            scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                        )
                    )
                    is_collision = c.in_collision_single(
                        crown_copy_2,
                        return_names=True,
                    )
                    if not is_collision[0]:
                        scale += 2
                        crown_copy_2 = crown_copy.copy()
                        crown_copy_2.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                            )
                        )
                        is_collision = c.in_collision_single(
                            crown_copy_2,
                            return_names=True,
                        )
                        if "mesh1" in list(is_collision[1]) and "mesh2" in list(
                            is_collision[1]
                        ):
                            crown.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[scale / 100, 1, 1],
                                    translate=[i / 100, 0, 0],
                                )
                            )
                            return crown
                    elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                        is_collision[1]
                    ):
                        for j in range(scale, 0, -2):
                            crown_copy_2 = crown_copy.copy()
                            crown_copy_2.apply_transform(
                                trimesh.transformations.compose_matrix(
                                    scale=[j / 100, 1, 1], translate=[i / 100, 0, 0]
                                )
                            )
                            is_collision = c.in_collision_single(
                                crown_copy_2,
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh2"]:
                                scale = j
                                break
                            elif list(is_collision[1]) == ["mesh1"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 2) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif not is_collision[0]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 2) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif list(is_collision[1]) == ["mesh1"]:
                        scale += 2
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1],
                                translate=[(i - 2.5) / 100, 0, 0],
                            )
                        )
                        return crown

        else:
            return crown  # 无碰撞，直接返回
    else:
        return crown  # 邻牙超过2个，暂不处理，直接返回


def adaptive_sparse_pointcloud_to_mesh(points, auto_detect_params=True):
    """
    自适应稀疏点云转网格，包含更详细的参数级别
    """
    # 分析点云密度

    nbrs = NearestNeighbors(n_neighbors=10).fit(points)
    distances, _ = nbrs.kneighbors(points)
    pc_points = points[np.where(np.mean(distances, axis=1) > 0.35)[0]]
    pc = trimesh.PointCloud(pc_points)
    return pc


def angle_between_vectors(a, b, ignore=-1):
    """
    计算两个向量之间的夹角(单位：弧度)
    """
    if ignore != -1:
        a[ignore] = b[ignore]
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    dot_product = np.dot(a, b)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)
    return angle_rad


def curvature_weights(pcd, percentile=90):
    # 计算曲率并生成权重（高曲率点权重=3）
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curv = []
    for i, pt in enumerate(pcd.points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pt, 0.06)
        if k < 3:
            curv.append(0)
            continue
        pts = np.asarray(pcd.points)[idx[1:], :]
        M = np.cov(pts - pts.mean(axis=0), rowvar=False)
        w, _ = np.linalg.eig(M)
        curv.append(w.min() / w.sum())
    curv = np.asarray(curv)
    w = np.ones_like(curv)
    w[curv > np.percentile(curv, percentile)] = 3.0
    return w


def preprocess(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd


def icp_criteria():
    return o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
    )


def symmetric_icp_with_weights(src, dst, weights, max_corr, init_T):
    # 把权重写进 src.colors[:,0]（Open3D 用颜色通道当权重）
    src.colors = o3d.utility.Vector3dVector(
        np.column_stack([weights, weights, weights])
    )
    reg = o3d.pipelines.registration.registration_generalized_icp(
        src,
        dst,
        max_corr,
        init_T,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=icp_criteria(),
    )
    return reg


def registration_with_scale(src_points, dst_points):
    src_orig = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_points))
    dst_orig = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_points))
    T_cumulative = np.eye(4)
    best_s = 1.0
    best_rmse = 1e9
    voxel_sizes = [0.1, 0.05, 0.01]
    for level, vs in enumerate(voxel_sizes):
        print(f"\n----- level {level + 1}  voxel={vs} -----")
        src = preprocess(src_orig, vs)
        dst = preprocess(dst_orig, vs)

        def scalar_rmse(s):
            src_s = copy.deepcopy(src)
            src_s.transform(np.diag([s, s, s, 1]))
            w = curvature_weights(src_s)
            reg = symmetric_icp_with_weights(src_s, dst, w, vs * 1.5, np.eye(4))
            if len(reg.correspondence_set) == 0:
                return np.inf
            return reg.inlier_rmse

        s_cand = np.linspace(0.7, 1.3, 31)  # 步长 0.02
        rmse_grid = np.array([scalar_rmse(s) for s in s_cand])
        rmse_grid = np.array([r if r != 0 else np.inf for r in rmse_grid])

        idx = np.argsort(rmse_grid)[0]
        rmse = rmse_grid[idx]
        s = s_cand[idx]
        if rmse < best_rmse and rmse != 0:
            best_s = s
            best_rmse = rmse
            print(f"  scale={best_s:.4f}  rmse={best_rmse:.4f}")
            src_scaled = copy.deepcopy(src)
            src_scaled.transform(np.diag([best_s, best_s, best_s, 1]))
            w = curvature_weights(src_scaled)
            reg = symmetric_icp_with_weights(src_scaled, dst, w, vs * 1.5, np.eye(4))
            if len(reg.correspondence_set) == 0:
                reg.inlier_rmse = np.inf
            best_T = reg.transformation
            T_cumulative = best_T @ T_cumulative
    return T_cumulative

def _nn_resample100(pts: np.ndarray):
    pts = np.asarray(pts)
    idx = np.round(np.linspace(0, len(pts) - 1, 100)).astype(int)
    return pts[idx]

def mirror_crown(
    all_other_crowns, beiya_id, crown_rot_matirx, ai_matrix, std_crown, mirror_id
):
    dac = DAC(all_other_crowns, None, int(beiya_id), 0)
    curve, _, control_points, T, A = dac.get_dac_nurbs()

    control_points = trimesh.PointCloud(control_points)
    control_points.apply_transform(np.linalg.pinv(T))
    # control_points.apply_transform(
    #     np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    # )
    # control_points.apply_transform(np.array(crown_rot_matirx[2]))
    # control_points.apply_transform(ai_matrix)

    bio_crown = read_mesh_bytes(all_other_crowns[mirror_id])
    # bio_crown.apply_transform(
    #     np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    # )
    # bio_crown.apply_transform(np.array(crown_rot_matirx[2]))
    # bio_crown.apply_transform(ai_matrix)

    n, d = get_mirror_plane(control_points[:50], control_points[50:][::-1])

    std_v, std_f = mirror_mesh(std_crown.vertices, std_crown.faces, n, d)
    std_mirror = trimesh.Trimesh(std_v, std_f)
    std_mirror.apply_translation(-std_mirror.bounding_box.centroid)
    std_mirror.apply_translation(bio_crown.bounding_box.centroid)

    bio_crown_morph, bio_crown_reg = bio_morphing(bio_crown, std_mirror, None, False)

    V_mir, F_mir = mirror_mesh(bio_crown_morph.vertices, bio_crown_morph.faces, n, d)

    bio_crown_mirror = trimesh.Trimesh(V_mir, F_mir)
    scale = (std_crown.bounds[1] - std_crown.bounds[0]) / (
        bio_crown_mirror.bounds[1] - bio_crown_mirror.bounds[0]
    )
    bio_crown_mirror.apply_translation(-bio_crown_mirror.bounding_box.centroid)
    bio_crown_mirror.apply_scale(scale)
    bio_crown_mirror.apply_translation(std_crown.bounding_box.centroid)

    other_teeth = [
        read_mesh_bytes(v) for k, v in all_other_crowns.items() if k != beiya_id
    ]
    bio_crown_mirror.apply_transform(np.array(crown_rot_matirx[0]))
    bio_crown_mirror.apply_transform(np.array(crown_rot_matirx[1]))
    bio_crown_mirror.apply_transform(np.array(crown_rot_matirx[2]))
    bio_crown_mirror.apply_transform(ai_matrix)
    for m in other_teeth:
        m.apply_transform(np.array(crown_rot_matirx[0]))
        m.apply_transform(np.array(crown_rot_matirx[1]))
        m.apply_transform(np.array(crown_rot_matirx[2]))
        m.apply_transform(ai_matrix)
    bio_crown_mirror = adjust_crown_position(bio_crown_mirror, other_teeth)
    bio_crown_mirror.apply_transform(np.linalg.pinv(ai_matrix))
    bio_crown_mirror.apply_transform(np.linalg.pinv(np.array(crown_rot_matirx[2])))
    bio_crown_mirror.apply_transform(np.linalg.pinv(np.array(crown_rot_matirx[1])))
    bio_crown_mirror.apply_transform(np.linalg.pinv(np.array(crown_rot_matirx[0])))

    return bio_crown_mirror


def mirror_crown_(
    arch_seg_crowns,
    all_other_crowns,
    beiya_id,
    crown_rot_matirx_mirror,
    crown_rot_matirx,
    ai_matrix,
    std_crown,
    mirror_id,
    prep_tid,
):
    dac = DAC(all_other_crowns, None, int(beiya_id), 0)
    curve, _, control_points, T, A = dac.get_dac_nurbs()

    control_points = trimesh.PointCloud(control_points)
    control_points.apply_transform(np.linalg.pinv(T))
    control_points.apply_transform(
        np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    )
    control_points.apply_transform(np.array(crown_rot_matirx[2]))
    control_points.apply_transform(ai_matrix)
    if len(control_points.vertices) != 100:
        control_points.vertices = _nn_resample100(control_points.vertices)
    n, d = get_mirror_plane(control_points[:50], control_points[50:][::-1])

    if (int(beiya_id) < 30 and int(mirror_id) > 30) or (
        int(beiya_id) > 30 and int(mirror_id) < 30
    ):
        dac = DAC(arch_seg_crowns, None, int(mirror_id), 0)
        curve, _, control_points_, T, A = dac.get_dac_nurbs()

        control_points_ = trimesh.PointCloud(control_points_)
        control_points_.apply_transform(np.linalg.pinv(T))
        control_points_.apply_transform(
            np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
        )
        control_points_.apply_transform(np.array(crown_rot_matirx[2]))
        control_points_.apply_transform(ai_matrix)
        if len(control_points_.vertices) != 100:
            control_points_.vertices = _nn_resample100(control_points_.vertices)
        n_, d_ = get_mirror_plane(control_points_[:50], control_points_[50:][::-1])
        c1 = np.mean((control_points_[:50] + control_points_[-50:]) / 2, axis=0)
        c2 = np.mean((control_points_[25:50] + control_points_[-50:-25]) / 2, axis=0)
        r_matrix = trimesh.transformations.rotation_matrix(np.pi, c1 - c2, c1)

    bio_crown = read_mesh_bytes(arch_seg_crowns[mirror_id])
    bio_crown.apply_transform(
        np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    )
    bio_crown.apply_transform(crown_rot_matirx_mirror[2])

    if int(beiya_id) < 30:
        if int(mirror_id) < 30:  # 同颌
            if int(beiya_id[0]) == int(mirror_id[0]):  # 同区
                V_mir, F_mir = bio_crown.vertices, bio_crown.faces
            else:  # 异区
                V_mir, F_mir = mirror_mesh(bio_crown.vertices, bio_crown.faces, n, d)
        else:  # 异颌
            if (int(beiya_id[0]) - int(mirror_id[0])) % 2 == 0:  # 不同侧
                bio_crown.apply_transform(r_matrix)
                V_mir, F_mir = bio_crown.vertices, bio_crown.faces
            else:  # 同侧
                bio_crown.apply_transform(r_matrix)
                V_mir, F_mir = mirror_mesh(bio_crown.vertices, bio_crown.faces, n_, d_)
    else:
        if int(mirror_id) < 30:  # 异颌
            if (int(beiya_id[0]) - int(mirror_id[0])) % 2 == 0:  # 不同侧
                bio_crown.apply_transform(r_matrix)
                V_mir, F_mir = bio_crown.vertices, bio_crown.faces
            else:  # 同侧
                bio_crown.apply_transform(r_matrix)
                V_mir, F_mir = mirror_mesh(bio_crown.vertices, bio_crown.faces, n_, d_)
        else:  # 同颌
            if int(beiya_id[0]) == int(mirror_id[0]):  # 同区
                V_mir, F_mir = bio_crown.vertices, bio_crown.faces
            else:  # 异区
                V_mir, F_mir = mirror_mesh(bio_crown.vertices, bio_crown.faces, n, d)

    std_crown.apply_transform(
        np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    )
    std_crown.apply_transform(np.array(crown_rot_matirx[2]))
    std_crown.apply_transform(ai_matrix)

    bio_crown = trimesh.Trimesh(V_mir, F_mir)
    if str(beiya_id)[1] in ["6", "7"]:
        if (
            int(beiya_id) < 30
            and int(mirror_id) < 30
            and int(beiya_id[1]) == int(mirror_id[1])
        ) or (
            int(beiya_id) > 30
            and int(mirror_id) > 30
            and int(beiya_id[1]) == int(mirror_id[1])
        ):
            pass
        else:
            bio_crown_box = bio_crown.bounding_box_oriented
            std_crown_box = std_crown.bounding_box_oriented
            bio_face_idx = np.argmax(bio_crown_box.face_normals[:, 1])
            std_face_idx = np.argmax(std_crown_box.face_normals[:, 1])
            bio_normal = np.array(bio_crown_box.face_normals[bio_face_idx])
            std_normal = np.array(std_crown_box.face_normals[std_face_idx])
            angle = angle_between_vectors(bio_normal, std_normal)
            bio_crown.apply_transform(
                trimesh.transformations.rotation_matrix(
                    angle, np.cross(bio_normal, std_normal), bio_crown.centroid
                )
            )
            bio_crown_box = bio_crown.bounding_box_oriented
            bio_normal = np.array(bio_crown_box.face_normals[bio_face_idx])
            angle_ = angle_between_vectors(bio_normal, std_normal)
            if min(angle_, np.pi - angle_) > min(angle, np.pi - angle):
                bio_crown.apply_transform(
                    trimesh.transformations.rotation_matrix(
                        -angle_, np.cross(bio_normal, std_normal), bio_crown.centroid
                    )
                )
    bio_crown.apply_translation(
        -(bio_crown.bounding_box.vertices[3] + bio_crown.bounding_box.vertices[7]) / 2
    )
    bio_crown.apply_translation(
        (std_crown.bounding_box.vertices[3] + std_crown.bounding_box.vertices[7]) / 2
    )

    bio_crown_morph, bio_crown_reg = bio_morphing(bio_crown, std_crown, None, False)

    if int(beiya_id) < 30:
        other_teeth_info = [
            (1, read_mesh_bytes(v)) if int(k) in prep_tid else (0, read_mesh_bytes(v))
            for k, v in all_other_crowns.items()
            if k != beiya_id and int(k) < 30
        ]
        other_teeth = [x[1] for x in other_teeth_info]
        beiya_tid = [x[0] for x in other_teeth_info]
    else:
        other_teeth_info = [
            (1, read_mesh_bytes(v)) if int(k) in prep_tid else (0, read_mesh_bytes(v))
            for k, v in all_other_crowns.items()
            if k != beiya_id and int(k) > 30
        ]
        other_teeth = [x[1] for x in other_teeth_info]
        beiya_tid = [x[0] for x in other_teeth_info]
    for m in other_teeth:
        m.apply_transform(np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1]))
        m.apply_transform(np.array(crown_rot_matirx[2]))
        m.apply_transform(ai_matrix)
    bio_crown_mirror = adjust_crown_position(bio_crown_morph, other_teeth, beiya_tid)
    bio_crown_mirror.apply_transform(np.linalg.pinv(ai_matrix))
    bio_crown_mirror.apply_transform(np.linalg.pinv(np.array(crown_rot_matirx[2])))
    combined_matrix = np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    bio_crown_mirror.apply_transform(np.linalg.inv(combined_matrix))
    return bio_crown_mirror
