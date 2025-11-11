import trimesh
from dental_arch_curve import DAC
import DracoPy
import numpy as np
import base64
import py_prepMorphing as morph
import random


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


def adjust_crown_position(crown, other_teeth):
    # 实现牙冠与其他牙齿的位置关系调整的逻辑
    crown_copy = crown.copy()
    crown_copy = crown_copy.simplify_quadratic_decimation(10000)
    adjacent_teeth = []
    c = trimesh.collision.CollisionManager()
    for tooth in other_teeth:
        dist = trimesh.base.proximity.closest_point(
            crown_copy, random.sample(tooth.vertices.tolist(), 100)
        )[1]
        if min(dist) < 0.5:  # 如果距离小于0.5mm，认为是邻牙
            adjacent_teeth.append(tooth)
    if len(adjacent_teeth) == 0:
        return crown  # 没有邻牙，直接返回
    elif len(adjacent_teeth) == 1:
        if adjacent_teeth[0].bounding_box.extents[0] > crown.bounding_box.extents[0]:
            c.add_object("mesh1", adjacent_teeth[0])
        else:
            c.add_object("mesh2", adjacent_teeth[0])
        is_collision = c.in_collision_single(
            crown,
            return_names=True,
        )
        if is_collision[0]:
            # 有碰撞，对牙冠进行缩小处理
            for i in range(100, 0, -5):
                is_collision = c.in_collision_single(
                    crown,
                    transform=trimesh.transformations.compose_matrix([i / 100, 1, 1]),
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
                is_collision = c.in_collision_single(
                    crown,
                    transform=trimesh.transformations.compose_matrix([i / 100, 1, 1]),
                    return_names=True,
                )
                if is_collision[0]:
                    crown.apply_transform(
                        trimesh.transformations.compose_matrix([i / 100, 1, 1])
                    )
                    return crown
    elif len(adjacent_teeth) == 2:
        if (
            adjacent_teeth[0].bounding_box.extents[0]
            > adjacent_teeth[1].bounding_box.extents[0]
        ):
            c.add_object("mesh1", adjacent_teeth[0])
            c.add_object("mesh2", adjacent_teeth[1])
        else:
            c.add_object("mesh1", adjacent_teeth[1])
            c.add_object("mesh2", adjacent_teeth[0])
        is_collision = c.in_collision_single(
            crown,
            return_names=True,
        )
        if is_collision[0]:
            if "mesh1" in list(is_collision[1]) and "mesh2" in list(is_collision[1]):
                # 同时与两个邻牙碰撞，对牙冠进行缩小处理
                shift = 0
                for i in range(100, 0, -5):
                    is_collision = c.in_collision_single(
                        crown,
                        transform=trimesh.transformations.compose_matrix(
                            scale=[i / 100, 1, 1],
                            translate=[shift / 100, 0, 0],
                        ),
                        return_names=True,
                    )
                    if list(is_collision[1]) == ["mesh1"]:
                        for j in range(shift, -200, -5):
                            is_collision = c.in_collision_single(
                                crown,
                                transform=trimesh.transformations.compose_matrix(
                                    scale=[i / 100, 1, 1],
                                    translate=[j / 100, 0, 0],
                                ),
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
                                        scale=[(i + 5) / 100, 1, 1],
                                        translate=[j / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif list(is_collision[1]) == ["mesh2"]:
                        for j in range(shift, 200, 5):
                            is_collision = c.in_collision_single(
                                crown,
                                transform=trimesh.transformations.compose_matrix(
                                    scale=[i / 100, 1, 1],
                                    translate=[j / 100, 0, 0],
                                    return_names=True,
                                ),
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
                                        scale=[(i + 5) / 100, 1, 1],
                                        translate=[j / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif not is_collision[0]:
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[(i + 5) / 100, 1, 1],
                                translate=[shift / 100, 0, 0],
                            )
                        )
                        return crown
            elif list(is_collision[1]) == ["mesh1"]:
                # 只与近中邻牙碰撞，对牙冠先平移后放缩
                scale = 100
                for i in range(0, -100, -5):
                    is_collision = c.in_collision_single(
                        crown,
                        transform=trimesh.transformations.compose_matrix(
                            scale=[scale / 100, 1, 1],
                            translate=[i / 100, 0, 0],
                        ),
                        return_names=True,
                    )
                    if list(is_collision[1]) == ["mesh2"]:
                        scale += 5
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1],
                                translate=[(i + 2.5) / 100, 0, 0],
                            )
                        )
                        return crown
                    elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                        is_collision[1]
                    ):
                        for j in range(scale, 0, -5):
                            is_collision = c.in_collision_single(
                                crown,
                                transform=trimesh.transformations.compose_matrix(
                                    scale=[j / 100, 1, 1], translate=[i / 100, 0, 0]
                                ),
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh1"]:
                                scale = j
                                break
                            elif list(is_collision[1]) == ["mesh2"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 5) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif not is_collision[0]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 5) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif not is_collision[0]:
                        scale += 5
                        is_collision = c.in_collision_single(
                            crown,
                            transform=trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                            ),
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
            elif list(is_collision[1]) == ["mesh2"]:
                # 只与远中邻牙碰撞，对牙冠先平移后放缩
                scale = 100
                for i in range(0, 100, 5):
                    is_collision = c.in_collision_single(
                        crown,
                        transform=trimesh.transformations.compose_matrix(
                            scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                        ),
                        return_names=True,
                    )
                    if list(is_collision[1]) == ["mesh1"]:
                        scale += 5
                        crown.apply_transform(
                            trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1],
                                translate=[(i - 2.5) / 100, 0, 0],
                            )
                        )
                        return crown
                    elif "mesh1" in list(is_collision[1]) and "mesh2" in list(
                        is_collision[1]
                    ):
                        for j in range(scale, 0, -5):
                            is_collision = c.in_collision_single(
                                crown,
                                transform=trimesh.transformations.compose_matrix(
                                    scale=[j / 100, 1, 1], translate=[i / 100, 0, 0]
                                ),
                                return_names=True,
                            )
                            if list(is_collision[1]) == ["mesh2"]:
                                scale = j
                                break
                            elif list(is_collision[1]) == ["mesh1"]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 5) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                            elif not is_collision[0]:
                                crown.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=[(j + 5) / 100, 1, 1],
                                        translate=[i / 100, 0, 0],
                                    )
                                )
                                return crown
                    elif not is_collision[0]:
                        scale += 5
                        is_collision = c.in_collision_single(
                            crown,
                            transform=trimesh.transformations.compose_matrix(
                                scale=[scale / 100, 1, 1], translate=[i / 100, 0, 0]
                            ),
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
    else:
        return crown  # 邻牙超过2个，暂不处理，直接返回


def mirror_crown(
    all_other_crowns, beiya_id, crown_rot_matirx, ai_matrix, std_crown, mirror_id
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

    bio_crown = read_mesh_bytes(all_other_crowns[mirror_id])
    bio_crown.apply_transform(
        np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    )
    bio_crown.apply_transform(np.array(crown_rot_matirx[2]))
    bio_crown.apply_transform(ai_matrix)

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
    bio_crown_mirror.apply_transform(
        np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1])
    )
    bio_crown_mirror.apply_transform(np.array(crown_rot_matirx[2]))
    bio_crown_mirror.apply_transform(ai_matrix)
    for m in other_teeth:
        m.apply_transform(np.array(crown_rot_matirx[0]) @ np.array(crown_rot_matirx[1]))
        m.apply_transform(np.array(crown_rot_matirx[2]))
        m.apply_transform(ai_matrix)
    bio_crown_mirror = adjust_crown_position(bio_crown_mirror, other_teeth)

    return bio_crown_mirror
