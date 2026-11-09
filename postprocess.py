import traceback

import numpy as np
import open3d
import requests
import trimesh

from inlay_cpu import InlayGeneration, MeshRegistration
from utils import (
    compress_drc,
    get_biggest_mesh,
    get_thickness_gap,
    o3d2tri,
    read_mesh_bytes,
    write_mesh_bytes,
)


def mesh_summary(mesh, label):
    if isinstance(mesh, trimesh.Trimesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    else:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
    if vertices.size == 0:
        return {"label": label, "verts": 0, "faces": 0}
    return {
        "label": label,
        "verts": int(len(vertices)),
        "faces": int(len(faces)),
        "bbox_min": np.round(vertices.min(axis=0), 4).tolist(),
        "bbox_max": np.round(vertices.max(axis=0), 4).tolist(),
        "center": np.round(vertices.mean(axis=0), 4).tolist(),
    }


def run(data):
    IG = InlayGeneration(
        tid=data.get("tid"),
        prep_tooth=data.get("prep_tooth"),
        inlay_inner=data.get("inlay_inner"),
        upper_scan=data.get("upper_scan"),
        lower_scan=data.get("lower_scan"),
        standard=data.get("standard"),
        fill_undercut=data.get("fill_undercut", False),
        paras=data.get("paras"),
        inner=data.get("inner"),
        thickness_shell=data.get("thickness_shell"),
    )
    IG.run()

    stitched_inlay, inlay_outer, inner_dilation, thickness_shell = (
        IG.get_stitched_inlay(),
        IG.get_inlay_outer(),
        IG.get_inlay_inner(),
        IG.get_thickness_shell(),
    )

    if not isinstance(stitched_inlay, trimesh.Trimesh):
        stitched_inlay = IG.o3d2tri(stitched_inlay)
    if not isinstance(inlay_outer, trimesh.Trimesh):
        inlay_outer = IG.o3d2tri(inlay_outer)
    if not isinstance(inner_dilation, trimesh.Trimesh):
        inner_dilation = IG.o3d2tri(inner_dilation)
    if not isinstance(thickness_shell, trimesh.Trimesh):
        thickness_shell = IG.o3d2tri(thickness_shell)

    return (
        stitched_inlay,
        inlay_outer,
        inner_dilation,
        thickness_shell,
        # IG.points_outer_id,
        IG.points_inner_id,
        IG.points_edge_outer_id,
        IG.points_edge_inner_id,
        IG.configs["mesh_tmp"],
    )


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
    data_input["prep_tooth"] = read_mesh_bytes(input_info.get("mesh_beiya"))
    data_input["inlay_inner"] = read_mesh_bytes(input_info.get("prep_q"))
    data_input["inlay_inner"] = get_biggest_mesh(data_input["inlay_inner"]).as_open3d
    data_input["upper_scan"] = read_mesh_bytes(input_info.get("mesh_upper"))
    data_input["lower_scan"] = read_mesh_bytes(input_info.get("mesh_lower"))
    data_input["standard"] = read_mesh_bytes(input_info.get("stdcrown"))
    # inner / thickness_shell: 由 thickness_shell 接口预计算后传入主流程；
    # 未传时 MeshRegistration.run 内回退到 get_cement_gap + get_thickness_gap（向后兼容）。
    if input_info.get("inner") is not None:
        data_input["inner"] = read_mesh_bytes(input_info.get("inner"))
    if input_info.get("thickness_shell") is not None:
        data_input["thickness_shell"] = read_mesh_bytes(input_info.get("thickness_shell"))
    data_input["tid"] = int(input_info.get("beiya_id"))
    paras = input_info.get("paras") or {}
    if input_info.get("cement_gap") is not None:
        paras["cement_gap"] = input_info.get("cement_gap")
    if input_info.get("insert_dir") is not None:
        paras["insert_dir"] = input_info.get("insert_dir")
    data_input["paras"] = paras
    print(
        f"decoded_input | {mesh_summary(data_input['inlay_inner'], 'prep_q_largest_component')}"
    )
    print(f"decoded_input | {mesh_summary(data_input['prep_tooth'], 'prep_tooth')}")
    mesh_tmp = {}
    if input_info.get("multi_restoration"):
        input_info["rot_matrix"] = input_info["rot_matrix"][0]
        input_info["rot_matrix"][0][-1][-1] = 1
        input_info["rot_matrix"][1][-1][-1] = 1

        data_input["prep_tooth"] = o3d2tri(data_input["prep_tooth"])
        data_input["prep_tooth"].apply_transform(input_info["rot_matrix"][0])
        data_input["prep_tooth"].apply_transform(input_info["rot_matrix"][1])
        data_input["prep_tooth"] = data_input["prep_tooth"].as_open3d

        data_input["inlay_inner"] = o3d2tri(data_input["inlay_inner"])
        data_input["inlay_inner"].apply_transform(input_info["rot_matrix"][0])
        data_input["inlay_inner"].apply_transform(input_info["rot_matrix"][1])
        data_input["inlay_inner"] = data_input["inlay_inner"].as_open3d

        data_input["upper_scan"] = o3d2tri(data_input["upper_scan"])
        data_input["upper_scan"].apply_transform(input_info["rot_matrix"][0])
        data_input["upper_scan"].apply_transform(input_info["rot_matrix"][1])
        data_input["upper_scan"] = data_input["upper_scan"].as_open3d

        data_input["lower_scan"] = o3d2tri(data_input["lower_scan"])
        data_input["lower_scan"].apply_transform(input_info["rot_matrix"][0])
        data_input["lower_scan"].apply_transform(input_info["rot_matrix"][1])
        data_input["lower_scan"] = data_input["lower_scan"].as_open3d

        data_input["standard"] = o3d2tri(data_input["standard"])
        data_input["standard"].apply_transform(input_info["rot_matrix"][0])
        data_input["standard"].apply_transform(input_info["rot_matrix"][1])
        data_input["standard"] = data_input["standard"].as_open3d

        if data_input.get("inner") is not None:
            data_input["inner"] = o3d2tri(data_input["inner"])
            data_input["inner"].apply_transform(input_info["rot_matrix"][0])
            data_input["inner"].apply_transform(input_info["rot_matrix"][1])
            data_input["inner"] = data_input["inner"].as_open3d

        if data_input.get("thickness_shell") is not None:
            data_input["thickness_shell"] = o3d2tri(data_input["thickness_shell"])
            data_input["thickness_shell"].apply_transform(input_info["rot_matrix"][0])
            data_input["thickness_shell"].apply_transform(input_info["rot_matrix"][1])
            data_input["thickness_shell"] = data_input["thickness_shell"].as_open3d

        data_input["fill_undercut"] = input_info.get("fill_undercut", False)

        po_out = run(data_input)

    if input_info.get("multi_restoration"):
        po_out[0].apply_transform(np.linalg.pinv(input_info["rot_matrix"][1]))
        po_out[0].apply_transform(np.linalg.pinv(input_info["rot_matrix"][0]))

        po_out[1].apply_transform(np.linalg.pinv(input_info["rot_matrix"][1]))
        po_out[1].apply_transform(np.linalg.pinv(input_info["rot_matrix"][0]))

        po_out[2].apply_transform(np.linalg.pinv(input_info["rot_matrix"][1]))
        po_out[2].apply_transform(np.linalg.pinv(input_info["rot_matrix"][0]))

        po_out[3].apply_transform(np.linalg.pinv(input_info["rot_matrix"][1]))
        po_out[3].apply_transform(np.linalg.pinv(input_info["rot_matrix"][0]))

        mesh_tmp = po_out[7]
        for k, v in mesh_tmp.items():
            v.apply_transform(np.linalg.pinv(input_info["rot_matrix"][1]))
            v.apply_transform(np.linalg.pinv(input_info["rot_matrix"][0]))
            mesh_tmp[k] = compress_drc(v)

    print("success postprocess")
    return {
        "crown": compress_drc(po_out[0], [po_out[4], po_out[5], po_out[6]]),
        "inlay_outer": write_mesh_bytes(po_out[1]),
        "inner_dilation": write_mesh_bytes(po_out[2]),
        "thickness_shell": write_mesh_bytes(po_out[3]),
        "mesh_tmp": mesh_tmp,
        "modal_function_call_id": None,
    }


def run_thickness_shell(input_info):
    """只跑 thickness_shell 生成：填倒凹 -> 粘接间隙 -> 厚度外壳。

    输入字段 inlay_inner（base64 Draco，与 postprocess 的 prep_q 同源）；
    始终执行填倒凹（undercut_py），再做 cement_gap 粘接间隙处理，最后 get_thickness_gap。
    不做 get_thickness_gap2 精修，无需 crown/upper/lower 等其它输入。

    可选输入：
      - insert_dir: 倒凹插入方向 [x, y, z]，未传时 undercut_py 自动搜索
      - cement_gap: 粘接间隙值，未传时用 configs.yaml 默认值 (0.08)；
        也可放在 paras.cement_gap 里，顶层 cement_gap 优先级更高
    """
    data_input = {}
    data_input["inlay_inner"] = read_mesh_bytes(input_info.get("inlay_inner"))
    data_input["inlay_inner"] = get_biggest_mesh(data_input["inlay_inner"]).as_open3d

    # cement_gap 覆盖：顶层 > paras > configs.yaml 默认
    paras = input_info.get("paras") or {}
    if input_info.get("cement_gap") is not None:
        paras["cement_gap"] = input_info.get("cement_gap")

    # tid=None: 仅借 InlayGeneration 加载 configs.yaml，不加载牙库、不设 prep/anta scan
    IG = InlayGeneration(
        tid=None,
        prep_tooth=None,
        inlay_inner=data_input["inlay_inner"],
        upper_scan=None,
        lower_scan=None,
        adjacent_teeth=None,
        standard=None,
        fill_undercut=False,
        paras=paras,
    )
    mr = MeshRegistration(IG.configs)
    # 1. 填倒凹（undercut_py：传入方向或自动搜索）— 复用 MeshRegistration.blockout_undercut
    insert_dir = input_info.get("insert_dir")
    IG.inlay_inner = mr.blockout_undercut(IG.inlay_inner, user_dir=insert_dir)
    # 2. 粘接间隙处理（cement_gap 膨胀 + TPS 重采样）
    # get_cement_gap 要求 trimesh 带 vertex_normals
    IG.inlay_inner.compute_vertex_normals()
    mr.get_cement_gap(mr.o3d2tri(IG.inlay_inner))  # 设 mr.inner_dilation (trimesh)
    # 3. 厚度外壳生成（get_thickness_gap 返回 (thickness_shell, inner)；
    #    inner 是供主流程复用的内表面，作为 postprocess 的输入回传）
    thickness_shell, inner = get_thickness_gap(mr.inner_dilation)

    if not isinstance(thickness_shell, trimesh.Trimesh):
        thickness_shell = IG.o3d2tri(thickness_shell)
    if not isinstance(inner, trimesh.Trimesh):
        inner = IG.o3d2tri(inner)

    print("success thickness_shell")
    return {
        "thickness_shell": write_mesh_bytes(thickness_shell),
        "inner": write_mesh_bytes(inner),
        "modal_function_call_id": None,
    }


def build_error_response(input_info):
    context_info = input_info if isinstance(input_info, dict) else {}
    res = {
        "error": traceback.format_exc(),
        "context": {
            "job_id": context_info.get("job_id"),
            "execution_id": context_info.get("execution_id"),
            "beiya_id": context_info.get("beiya_id"),
            "multi_restoration": context_info.get("multi_restoration", False),
            "fill_undercut": context_info.get("fill_undercut", False),
            "paras": context_info.get("paras"),
        },
        "modal_function_call_id": None,
    }
    print(f"postprocess_error_context: {res['context']}")
    return res


def handler(event, context=None, requests_module=None):
    requests_module = requests if requests_module is None else requests_module
    input_info = event
    print("receive case")
    try:
        print("start AI_Inlay_Post ..")
        input_info = load_input_info(event, requests_module)
        input_info = normalize_input(input_info)
        log_request_info(input_info)
        action = input_info.get("action")
        if action == "thickness_shell":
            result_json = run_thickness_shell(input_info)
        else:
            result_json = run_pipeline(input_info)
        if event.get("target_url"):
            upload_result(event["target_url"], result_json, requests_module)
            return success_response("result saved to target url")
        return success_response(result_json)
    except Exception as _:
        res = build_error_response(input_info)
        traceback.print_exc()
        return res


if __name__ == "__main__":
    pass
