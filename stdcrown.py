import base64
import json
import traceback

import DracoPy
import numpy as np
import trimesh
from scipy.spatial import transform
from pre_op_mirror import pre_op, mirror_crown
from utils import get_biggest_mesh

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


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


def run(data):
    std_crown = read_mesh_bytes(data["std_crown"])
    
    if data["preop_or_mirror"] == "preop":
        pre_op_crown = read_mesh_bytes(data["pre_op_crown"])
        preop_matrix = data["preop_matrix"]
        # pre_op_teeth = read_mesh_bytes(data["pre_op_teeth"])
        # post_op_teeth = read_mesh_bytes(data["post_op_teeth"])

        post_op_crown = pre_op.std_morphing(std_crown, pre_op_crown)
        post_op_crown.apply_transform(preop_matrix)

        # post_op_crown = pre_op.preop2post(pre_op_teeth.as_open3d, post_op_teeth.as_open3d, pre_op_morph.as_open3d)
        # post_op_crown = trimesh.Trimesh(post_op_crown.vertices, post_op_crown.triangles)
        return post_op_crown
    elif data["preop_or_mirror"] == "mirror":
        m_crown = mirror_crown.mirror_crown(data["all_other_crown"], data["beiya_id"], data["rot_matrix"], np.eye(4), std_crown, data["mirror_id"])
        if data["test"]:
            save_path = data["save_path"]
            m_crown.export(f"{save_path}/stdcrown_mirror_{data['beiya_id']}.stl")
        return m_crown

def remesh(mesh):
    import pymeshlab
    ms_mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(ms_mesh)
    ms.meshing_isotropic_explicit_remeshing()

    return trimesh.Trimesh(ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix())


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_CPU_Std ..")
        if event.get("job_id"):
            print(f"job_id: {event.get('job_id')}")
        else:
            print("job_id: None")
        if event.get("execution_id"):
            print(f"execution_id: {event.get('execution_id')}")
        else:
            print("execution_id: None")
        data_input = {}
        data_input["beiya_id"] = event.get("beiya_id")
        data_input["preop_or_mirror"] = event.get("preop_or_mirror")
        data_input["std_crown"] = event.get("std_crown")
        data_input["pre_op_crown"] = event.get("pre_op_crown")
        data_input["preop_matrix"] = event.get("preop_matrix", np.eye(4))
        # data_input["pre_op_teeth"] = event.get("pre_op_teeth")
        data_input["mirror_id"] = event.get("mirror_id")
        if data_input["beiya_id"][0] in ["1", "2"]:
            data_input["all_other_crown"] = event.get("all_other_crown")["upper_arch"].get('teeth_crowns')
        else:
            data_input["all_other_crown"] = event.get("all_other_crown")["lower_arch"].get('teeth_crowns')
        data_input["rot_matrix"] = event["rot_matrix"][0]
        data_input["rot_matrix"][0][-1][-1] = 1
        data_input["rot_matrix"][1][-1][-1] = 1
        data_input["rot_matrix"].append(np.eye(4).tolist())
        data_input["test"] = event.get("test", False)
        data_input["save_path"] = event.get("save_path", "")

        std_out = run(data_input)

        std_out = remesh(std_out)
        std_out = get_biggest_mesh(std_out)

        std_json = {
            "stdcrown": write_mesh_bytes(std_out),
        }
        print("suncess stitch_edge")

        return {"Msg": {"data": std_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {
            "error": traceback.format_exc(),
            "modal_function_call_id": None,
        }
        traceback.print_exc()
        return res


if __name__ == "__main__":
    # with open("test_data/c1/inlay_standard_lambda.json", "r") as f:
    #     data_gpu = json.load(f)
    with open("test_data/c1/inlay_standard_lambda.json", "r") as f:
        data = json.load(f)
    
    # event = {
    #     "beiya_id": 45,
    #     "preop_or_mirror": "mirror",
    #     "std_crown": data["stdcrown"],
    #     "mirror_id": "34",
    #     "all_other_crown": data_gpu["all_teeth_seg"],
    #     "rot_matrix": data_gpu["inlay_res"]["45"]["rot_matrix"]
    # }
    data["test"] = True
    data["save_path"] = "test_data/c1"
    handler(data, None)