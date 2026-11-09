import base64
import json
import traceback

import DracoPy
import numpy as np
import requests
import trimesh
from scipy.spatial import transform
from pre_op_mirror import pre_op, mirror_crown
from utils import get_biggest_mesh, transform_method_
from scipy.spatial.distance import cdist

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

def getBoundaryPoints(mesh):
    """get the boundary points of a mesh IN THE ORDER of how they appear in the mesh"""
    unique_elements, counts = np.unique(mesh.edges_sorted, return_counts=True, axis=0)
    edges = unique_elements[np.where(counts == 1)[0]]
    boundary_point_id = np.unique(edges.flatten())
    boundary_point = mesh.vertices[boundary_point_id]
    return boundary_point, boundary_point_id



def run(data):
    std_crown = read_mesh_bytes(data["std_crown"])
    
    if data["preop_or_mirror"] == "preop":
        pre_op_crown = read_mesh_bytes(data["pre_op_crown"])
        preop_matrix = data["preop_matrix"]
        # pre_op_teeth = read_mesh_bytes(data["pre_op_teeth"])
        # post_op_teeth = read_mesh_bytes(data["post_op_teeth"])

        pre_op_crown.apply_transform(preop_matrix)
        b_p, _ = getBoundaryPoints(pre_op_crown)
        post_op_crown = pre_op.std_morphing(std_crown, pre_op_crown)
        dist = cdist(b_p, post_op_crown.vertices)
        b_p_id = np.unique(np.where(dist < 1)[1]).tolist()
        post_op_crown = trimesh.smoothing.filter_laplacian(post_op_crown, laplacian_operator=trimesh.smoothing.laplacian_calculation(post_op_crown, pinned_vertices=np.setdiff1d(np.arange(post_op_crown.vertices.shape[0]), b_p_id)))
        
        return post_op_crown
    elif data["preop_or_mirror"] == "mirror":
        # m_crown = mirror_crown.mirror_crown(data["all_other_crown"], data["beiya_id"], data["rot_matrix"], np.eye(4), std_crown, data["mirror_id"])
        _, mirror_matrix = transform_method_(read_mesh_bytes(data['arch_seg_crowns'][data['mirror_id']]),
                data['arch_seg_crowns'],
                data['mirror_id'],
                0,
                None,
                None,)
        m_crown = mirror_crown.mirror_crown_(
            data['arch_seg_crowns'],
            data['all_other_crown'],
            data['beiya_id'],
            mirror_matrix,
            data['rot_matrix'],
            np.eye(4),
            std_crown.copy(),
            data['mirror_id'],
            data['prep_tid'],
        )
        if data["test"]:
            save_path = data["save_path"]
            m_crown.export(f"{save_path}/stdcrown_mirror_{data['beiya_id']}.stl")
        return m_crown

def remesh(mesh):
    import pymeshlab
    ms_mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(ms_mesh)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(0.15), featuredeg=10, )

    return trimesh.Trimesh(ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix())


def load_input_info(event, requests_module=None):
    requests_module = requests if requests_module is None else requests_module
    if event.get("source_url"):
        return requests_module.get(event["source_url"]).json()
    return event


def upload_result(target_url, data, requests_module=None):
    requests_module = requests if requests_module is None else requests_module
    response = requests_module.put(
        target_url,
        json=data,
        headers={"Content-Type": "binary/octet-stream"},
    )
    response.raise_for_status()
    return response


def normalize_input(input_info):
    return input_info


def success_response(data):
    return {"Msg": {"data": data}, "Code": 200, "State": "Success"}


def log_request_info(input_info):
    if input_info.get("job_id"):
        print(f"job_id: {input_info.get('job_id')}")
    else:
        print("job_id: None")
    if input_info.get("execution_id"):
        print(f"execution_id: {input_info.get('execution_id')}")
    else:
        print("execution_id: None")


def run_pipeline(input_info):
    data_input = {}
    data_input["beiya_id"] = input_info.get("beiya_id")
    data_input["preop_or_mirror"] = input_info.get("preop_or_mirror")
    data_input["std_crown"] = input_info.get("std_crown")
    data_input["pre_op_crown"] = input_info.get("pre_op_crown")
    # data_input["pre_op_teeth"] = input_info.get("pre_op_teeth")
    data_input["mirror_id"] = input_info.get("mirror_id")
    if data_input["preop_or_mirror"] == "mirror":
        arch_seg_crowns = {}
        if data_input["beiya_id"][0] in ["1", "2"]:
            data_input["all_other_crown"] = input_info.get("all_other_crown")["upper_arch"].get("teeth_crowns")
        else:
            data_input["all_other_crown"] = input_info.get("all_other_crown")["lower_arch"].get("teeth_crowns")
        if data_input["mirror_id"][0] in ["1", "2"]:
            for k, v in input_info.get("all_other_crown")["upper_arch"].items():
                arch_seg_crowns = {**arch_seg_crowns, **v}
        else:
            for k, v in input_info.get("all_other_crown")["lower_arch"].items():
                arch_seg_crowns = {**arch_seg_crowns, **v}
        data_input["arch_seg_crowns"] = arch_seg_crowns
        data_input["prep_tid"] = input_info.get("prep_tid")
        data_input["rot_matrix"] = input_info["rot_matrix"][0]
        data_input["rot_matrix"][0][-1][-1] = 1
        data_input["rot_matrix"][1][-1][-1] = 1
        data_input["rot_matrix"].append(np.eye(4).tolist())
    else:
        if data_input["beiya_id"][0] in ["1", "2"]:
            data_input["preop_matrix"] = input_info.get("preop_to_postop").get("upper_matrix", np.eye(4))
        else:
            data_input["preop_matrix"] = input_info.get("preop_to_postop").get("lower_matrix", np.eye(4))
    data_input["test"] = input_info.get("test", False)
    data_input["save_path"] = input_info.get("save_path", "")

    std_out = run(data_input)
    std_out = remesh(std_out)
    std_out = get_biggest_mesh(std_out)
    print("suncess stitch_edge")
    return {"stdcrown": write_mesh_bytes(std_out)}


def handler(event, context=None, requests_module=None):
    requests_module = requests if requests_module is None else requests_module
    input_info = event
    print("receive case")
    try:
        print("start AI_Inlay_CPU_Std ..")
        input_info = load_input_info(event, requests_module)
        input_info = normalize_input(input_info)
        log_request_info(input_info)
        std_json = run_pipeline(input_info)
        if event.get("target_url"):
            upload_result(event["target_url"], std_json, requests_module)
            return success_response("result saved to target url")
        return success_response(std_json)
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
    with open("/home/wanglong/下载/data/02/inlay_standard_lambda.json", "r") as f:
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
    data["save_path"] = "."
    handler(data, None)
