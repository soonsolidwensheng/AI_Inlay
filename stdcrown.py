import base64
import json
import traceback

import DracoPy
import numpy as np
import trimesh
from scipy.spatial import transform


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
    return trimesh.Trimesh(V, F).as_open3d


def run(data):
    with open("config/st_tooth_remesh/config.json", "r") as f:
        tooth_lib_configs = json.load(f)[str(data["beiya_id"])]
    ori_mesh = trimesh.load(tooth_lib_configs["st_path"])
    ori_mesh.apply_translation(-ori_mesh.centroid)
    points_bottom = tooth_lib_configs["cross_points"][1]
    points_top = tooth_lib_configs["cross_points"][0]
    points_front = tooth_lib_configs["cross_points"][2]
    points_back = tooth_lib_configs["cross_points"][3]

    v_occl = ori_mesh.vertices[points_top] - ori_mesh.vertices[points_bottom]
    v_misial = ori_mesh.vertices[points_front] - ori_mesh.vertices[points_back]
    frameA = np.array([v_occl, v_misial])
    frameB = np.array([[0, 1, 0], [1, 0, 0]])
    weights = np.array([0.5, 0.5])
    rot_mat, root_sum_squared_distance = transform.Rotation.align_vectors(
        frameA, frameB, weights=weights
    )
    rot_matrix = np.eye(4)  # 创建一个单位矩阵作为变换矩阵的初始值
    rot_matrix[:3, :3] = rot_mat.as_matrix()  # 复制旋转矩阵的前三列到变换矩阵的前三列
    rot_matrix[:, 3] = [0, 0, 0, 1]
    ori_mesh.apply_transform(np.linalg.pinv(rot_matrix))

    return ori_mesh


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
        std_out = run(data_input)

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
    event = {
        "beiya_id": 16,
        "job_id": "123",
        "execution_id": "456",
    }
    print(handler(event, None))
