import base64
import traceback
import DracoPy
import numpy as np
import trimesh
import json
from crown_cpu import occlu
import time
import sys
import MQCompressPy


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


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
    # encoding_test = DracoPy.encode_mesh_to_buffer(mesh.vertices, mesh.faces, preserve_order=preserve_order)
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def write_drc(drc, file):
    with open(file, "wb") as test_file:
        test_file.write(base64.b64decode(drc))


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

def file_name2template_name(file_name):
    s = file_name.split('_')[0]
    f2t = {
        'CSv1': 'Cyber Standard v1',
        'GSv1': 'Generic Standard v1',
        'Mv1': 'Mature v1',
        'rev1': 'st_tooth',
        'SSv1': 'Soft Standard v1',
        'Yv1': 'Youth v1'
        }
    return f2t[s]

def handler(event, context):
    print("receive case")
    try:
        print("start AI_Crown_Occlu ..")

        # 本地rie测试输入
        # event["paras"] = {
        #     "ad_gap": 0,
        #     "occlusal_distance": 0,
        #     "prox_or_occlu": 3,
        #     "thick_flag": 0,
        #     "approximal_distance": None,
        # }

        # event["trans_matrix"] = [
        #     [0.96593, -0.25882, 0, 0],
        #     [0.25882,  0.96593, 0, 0],
        #     [0,        0,       1, 0],
        #     [0,        0,       0, 1]
        # ]

        # event['out'] = event['fixed_out']
        print("pred_filestem_name", event.get("pred_filestem_name"))
        if event.get("pred_filestem_name"):
            event["template_name"] = file_name2template_name(event.get("pred_filestem_name"))
        print("event keys", event.keys())
        cpu_input_json = {
            "out": event.get("out"),
            "inner": event.get("inner"),
            "paras": event.get("paras"),
            "trans_matrix": event.get("trans_matrix"),
        }
        for k, v in event["cpu_info_json"].items():
            event[k] = v
        print("template_name", event.get("template_name"))
        print("job_id", event.get("job_id"))
        print("cpu_info_json keys", event["cpu_info_json"].keys())
        print("linya_points", event["cpu_info_json"].get("linya_points"))
        occlu_out = occlu(event)
        inner = write_mesh_bytes(occlu_out.mesh_beiya)
        if len(occlu_out.thickness_face_id):
            if occlu_out.thick_flag:
                crown = write_mesh_bytes(occlu_out.mesh_without_thickness)
                out = write_mesh_bytes(occlu_out.mesh_outside_without_thickness)
                fixed_points = np.array(occlu_out.thickness_points_id).tolist()
                fixed_crown = compress_drc(occlu_out.mesh, fixed_points)
                fixed_out = write_mesh_bytes(occlu_out.mesh_outside)
                without_thickness_add_points = np.array(
                    occlu_out.without_thickness_add_points
                ).tolist()
                without_thickness_add_point_normal = np.array(
                    occlu_out.without_thickness_add_point_normal
                ).tolist()
                without_thickness_cross_points = np.array(
                    occlu_out.without_thickness_cross_points
                ).tolist()
                without_thickness_adj_points1 = np.array(
                    occlu_out.without_thickness_adj_points1
                ).tolist()
                without_thickness_adj_points2 = np.array(
                    occlu_out.without_thickness_add_points
                ).tolist()
                without_thickness_print_points = (
                    trimesh.PointCloud(without_thickness_add_points)
                    .apply_transform(occlu_out.print_matrix)
                    .vertices.tolist()
                )
                without_thickness_print_normal = (
                    trimesh.PointCloud(without_thickness_add_point_normal)
                    .apply_transform(occlu_out.print_matrix)
                    .vertices.tolist()
                )
            else:
                fixed_points = np.array(occlu_out.thickness_points_id).tolist()
                crown = compress_drc(occlu_out.mesh, fixed_points)
                out = write_mesh_bytes(occlu_out.mesh_outside)
                fixed_crown = ""
                fixed_out = ""
                fixed_points = []
                without_thickness_add_points = []
                without_thickness_add_point_normal = []
                without_thickness_print_points = []
                without_thickness_print_normal = []
                without_thickness_cross_points = []
                without_thickness_adj_points1 = []
                without_thickness_adj_points2 = []
        else:
            crown = write_mesh_bytes(occlu_out.mesh)
            out = write_mesh_bytes(occlu_out.mesh_outside)
            fixed_crown = ""
            fixed_out = ""
            fixed_points = []
            without_thickness_add_points = []
            without_thickness_add_point_normal = []
            without_thickness_print_points = []
            without_thickness_print_normal = []
            without_thickness_cross_points = []
            without_thickness_adj_points1 = []
            without_thickness_adj_points2 = []
        add_points = np.array(occlu_out.mesh.ad_points.pt).tolist()
        add_point_normal = occlu_out.add_point_normal.tolist()
        print_points = (
            trimesh.PointCloud(add_points)
            .apply_transform(occlu_out.print_matrix)
            .vertices.tolist()
        )
        print_normal = (
            trimesh.PointCloud(add_point_normal)
            .apply_transform(occlu_out.print_matrix)
            .vertices.tolist()
        )
        axis = np.array(occlu_out.axis).tolist()
        print_matrix = np.array(occlu_out.print_matrix).tolist()
        miss_id = occlu_out.miss_id
        adj_points1 = np.array(occlu_out.mesh.adj_points1.pt).tolist()
        adj_points2 = np.array(occlu_out.mesh.adj_points2.pt).tolist()
        cross_points = np.array(occlu_out.mesh.cross_points.pt).tolist()
        if "linya_points" in vars(occlu_out.mesh):
            linya_points = np.array(occlu_out.mesh.linya_points.pt).tolist()
        else:
            linya_points = None
        # occlu_out.mesh_outside_without_thickness.export("mesh_owt.stl")
        # occlu_out.mesh_without_thickness.export("mesh_wt.stl")
        # occlu_out.mesh_outside.export("mesh_ot.stl")
        # occlu_out.mesh.export("mesh_t.stl")
        # occlu_out.mesh_beiya.export("beiya.stl")

        cpu_info_json = {
            "cpu_points_info": {
                "ad_points": add_points,
                "cross_points": cross_points,
                "adj_points1": adj_points1,
                "adj_points2": adj_points2,
                "linya_points": linya_points,
            },
            "beiya_id": miss_id,
            "axis": axis,
            "closer": write_mesh_bytes(occlu_out.mesh1),
            "further": write_mesh_bytes(occlu_out.mesh2),
            "is_single": occlu_out.is_single,
            "mesh_oppo": write_mesh_bytes(occlu_out.mesh_oppo),
        }
        cpu_info_json_without_thickness = {
            "cpu_points_info": {
                "ad_points": without_thickness_add_points,
                "cross_points": without_thickness_cross_points,
                "adj_points1": without_thickness_adj_points1,
                "adj_points2": without_thickness_adj_points2,
                "linya_points": linya_points,
            },
            "beiya_id": miss_id,
            "axis": axis,
            "closer": write_mesh_bytes(occlu_out.mesh1),
            "further": write_mesh_bytes(occlu_out.mesh2),
            "is_single": occlu_out.is_single,
            "mesh_oppo": write_mesh_bytes(occlu_out.mesh_oppo),
        }
        occlu_json = {
            "crown": crown,
            "out": out,
            "inner": inner,
            "fixed_crown": fixed_crown,
            "fixed_out": fixed_out,
            "points_info": {
                "points": without_thickness_print_points,
                "normals": without_thickness_print_normal,
                "matrix": print_matrix,
            },
            "fixed_points_info": {
                "points": print_points,
                "normals": print_normal,
                "matrix": print_matrix,
            },
            "cpu_info_json": cpu_info_json_without_thickness,
            "fixed_cpu_info_json": cpu_info_json,
            "cpu_input_json": cpu_input_json,
        }

        # 本地rie测试保存结果
        # with open("occlu0828.json", "w") as f:
        #     f.write(json.dumps(occlu_json))

        print("suncess occlu")
        occlu_json_size = sys.getsizeof(json.dumps(occlu_json))
        print(f"Size of occlu_json: {occlu_json_size} bytes")
        return {"Msg": {"data": occlu_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {"Msg": traceback.format_exc(), "Code": 203, "State": "Failure"}
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import time
    import os

    # save_dir = r"test_data_"
    # case_id = "3d4c"
    # with open(os.path.join(save_dir, case_id, "output.json"), "r") as f:
    #     event = json.load(f)
    # s1 = time.time()
    # # event["paras"] = {
    # #     "occlusal_distance": 0.3,
    # #     "ad_gap": 0,
    # #     "prox_or_occlu": 3,
    # #     "align_edges": True,
    # # }
    # out = handler(event, os.path.join(save_dir, case_id))
    # s2 = time.time()
    # print(s2 - s1)
    # with open(os.path.join(save_dir, case_id, "occlu_.json"), "w") as f:
    #     f.write(json.dumps(out["Msg"]["data"]))
    