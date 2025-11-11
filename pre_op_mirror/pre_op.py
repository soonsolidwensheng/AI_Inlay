import numpy as np
import py_prepMorphing as morph
import trimesh
import open3d
import pypreopcrowner
from scipy.spatial.transform import Rotation as sr


def std_morphing(std_crown, pre_crown):
    """
    Perform standard morphing between a standard crown and a pre-crown mesh.

    Args:
        std_crown (trimesh.Trimesh): The standard crown mesh.
        pre_crown (trimesh.Trimesh): The pre-crown mesh.
    """
    V_template = std_crown.vertices.astype(np.float64)
    F_template = std_crown.faces.astype(np.int32)
    V_to = pre_crown.vertices.astype(np.float64)
    F_to = pre_crown.faces.astype(np.int32)

    dstOrient = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    success, V_reg, F_reg = morph.GeneralRegistration(
        V_template, F_template, V_to, F_to, 50, dstOrient, True
    )
    print("GeneralRegistration success:", success)
    success, V_rebuild, F_rebuild = morph.ItersMorphingAndRebuild(
        V_reg,
        F_reg,
        V_to,
        F_to,
        isRemoveBoundary=True,
    )
    print("ItersMorphingAndRebuild success:", success)
    return trimesh.Trimesh(vertices=V_rebuild, faces=F_rebuild)


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=30)
    )
    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10.0, max_nn=100),
    )
    return (pcd_down, pcd_fpfh)


def VF_to_pypreopcrowner(V, F) -> pypreopcrowner.AlgorithmMesh:
    mesh_pypreopcrowner = pypreopcrowner.AlgorithmMesh()
    mesh_pypreopcrowner.vertices = np.asarray(V)
    mesh_pypreopcrowner.faces = np.asarray(F)
    return mesh_pypreopcrowner


def scan_register(pre_v, pre_f, post_v, post_f):
    """用source mesh 去匹配target mesh, 返回配准后的reg_source"""
    algo = pypreopcrowner.PreopcrownerSharedInterface.createObject()
    # read mesh

    pre_mesh = VF_to_pypreopcrowner(pre_v, pre_f)
    post_mesh = VF_to_pypreopcrowner(post_v, post_f)
    # additional tooth
    # source_tooth_mesh = VF_to_pypreopcrowner(source_tooth_v, source_tooth_f)
    # set parameters
    reg_param = pypreopcrowner.RegistrationParam()  # reserve, unused
    reg_out = pypreopcrowner.RegistrationOutput()
    # run
    success = algo.preop_registration(post_mesh, pre_mesh, reg_param, reg_out)
    # get the results
    if success:
        print("Registration success.")
        # print("rotation(quaternion#xyzw): ", reg_out.rotation[0], reg_out.rotation[1], reg_out.rotation[2], reg_out.rotation[3])
        # print("translation(vec#xyz): ", reg_out.translation[0], reg_out.translation[1], reg_out.translation[2])

        R = sr.from_quat(
            [
                reg_out.rotation[0],
                reg_out.rotation[1],
                reg_out.rotation[2],
                reg_out.rotation[3],
            ]
        )
        T = [reg_out.translation[0], reg_out.translation[1], reg_out.translation[2]]
        # ...
        registered_verts = R.apply(post_mesh.vertices) + T

        # #### rotate source tooth
        # registered_tooth_verts = R.apply(source_tooth_mesh.vertices) + T

        return registered_verts, post_mesh.faces, R.as_matrix(), T


def preop2post(preop_teeth_o3d, post_teeth_o3d, preop_crown_o3d):
    """
    preop_teeth_o3d: 术前牙颌
    post_teeth_o3d: 术后牙颌
    preop_crown_o3d: 术前牙冠
    """
    preop_teeth_pc = open3d.geometry.PointCloud(preop_teeth_o3d.vertices)
    post_teeth_pc = open3d.geometry.PointCloud(post_teeth_o3d.vertices)
    voxel_size = 0.25
    distance_threshold = 10 * voxel_size
    preop_teeth_pc_down, preop_teeth_pc_fpfh = preprocess_point_cloud(
        preop_teeth_pc, voxel_size
    )
    post_teeth_pc_down, post_teeth_pc_fpfh = preprocess_point_cloud(
        post_teeth_pc, voxel_size
    )
    global_reg = (
        open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            preop_teeth_pc_down,  # source pc
            post_teeth_pc_down,  # destination pc
            preop_teeth_pc_fpfh,  # source feature
            post_teeth_pc_fpfh,  # destination feature
            open3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold,
                iteration_number=64,
                maximum_tuple_count=1000,
            ),
        )
    )
    preop_teeth_o3d.transform(global_reg.transformation)
    preop_crown_o3d.transform(global_reg.transformation)

    V, F, R, T = scan_register(
        np.asarray(preop_teeth_o3d.vertices),
        np.asarray(preop_teeth_o3d.triangles),
        np.asarray(post_teeth_o3d.vertices),
        np.asarray(post_teeth_o3d.triangles),
    )


    ICP_trans_mat = np.eye(4)
    ICP_trans_mat[:3, :3] = R
    ICP_trans_mat[:3, 3] = T
    preop_reg_trans_mat = ICP_trans_mat @ global_reg.transformation

    # write mesh
    prep_op_teeth_registered = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(V), open3d.utility.Vector3iVector(F)
    )
    prep_op_teeth_registered.compute_vertex_normals()

    #! write mesh for prep op tooth
    prep_op_v_homogeneous = np.hstack(
        [
            np.asarray(preop_crown_o3d.vertices),
            np.ones((np.asarray(preop_crown_o3d.vertices).shape[0], 1)),
        ]
    )
    transformed_prep_op_v_homogeneous = prep_op_v_homogeneous @ preop_reg_trans_mat.T
    prep_op_v_new = transformed_prep_op_v_homogeneous[:, :3]


    prep_op_tooth_registered = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(prep_op_v_new),
        open3d.utility.Vector3iVector((np.asarray(preop_crown_o3d.triangles))),
    )
    prep_op_tooth_registered.compute_vertex_normals()

    return preop_crown_o3d
