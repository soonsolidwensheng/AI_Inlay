import base64
import traceback

import DracoPy
import numpy as np
import trimesh

from inlay_cpu import InlayGeneration, MeshRegistration
from utils import compress_drc, o3d2tri
from undercut_util import get_insert_direction_copy
from directional_undercut_filling import filling_undercut


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
    IG = InlayGeneration(
        tid=None,
        prep_tooth=None,
        inlay_inner=data.get("inner_dilation"),
        upper_scan=None,
        lower_scan=None,
        adjacent_teeth=None,
        standard=data.get("inlay_outer"),
        fill_undercut=data.get("fill_undercut", False)
    )
    mesh_registration = MeshRegistration(IG.configs)
    if mesh_registration.fill_undercut:
        insert_direction = get_insert_direction_copy(IG.inlay_inner)
        IG.inlay_inner = filling_undercut(IG.inlay_inner, insert_direction, display=False)[0].as_open3d
    mesh_registration.get_cement_gap(IG.o3d2tri(IG.inlay_inner))
    # boundarySet, _ = mesh_registration.getBoundaryPoints(mesh_registration.inner_dilation.as_open3d)
    # IG.lib_tooth = mesh_registration.doSubMeshTps(boundarySet, IG.lib_tooth, mesh_registration.lastSubmehs_tps_dis)
    IG.inner_dilation = mesh_registration.inner_dilation
    IG.inlay_outer = IG.lib_tooth
    # IG.stitch()
    # IG.get_inner_outer_edge_idx()
    IG.stitch_new()

    return (
        IG.o3d2tri(IG.get_stitched_inlay()),
        IG.inner_dilation,  
        # IG.points_outer_id,
        IG.points_inner_id,
        IG.points_edge_outer_id,
        IG.points_edge_inner_id,
    )


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_Stitch_Edge ..")
        if event.get("job_id"):
            print(f"job_id: {event.get('job_id')}")
        else:
            print("job_id: None")
        if event.get("execution_id"):
            print(f"execution_id: {event.get('execution_id')}")
        else:
            print("execution_id: None")
        data_input = {}
        data_input["inner_dilation"] = read_mesh_bytes(event.get("inner_dilation"))
        data_input["inlay_outer"] = read_mesh_bytes(event.get("inlay_outer"))
        if event.get("multi_restoration"):
            event["rot_matrix"] = event["rot_matrix"][0]
            event["rot_matrix"][0][-1][-1] = 1
            event["rot_matrix"][1][-1][-1] = 1
            data_input["inner_dilation"] = o3d2tri(data_input["inner_dilation"])
            data_input["inner_dilation"].apply_transform(event["rot_matrix"][0])
            data_input["inner_dilation"].apply_transform(event["rot_matrix"][1])
            data_input["inner_dilation"] = data_input["inner_dilation"].as_open3d

            data_input["inlay_outer"] = o3d2tri(data_input["inlay_outer"])
            data_input["inlay_outer"].apply_transform(event["rot_matrix"][0])
            data_input["inlay_outer"].apply_transform(event["rot_matrix"][1])
            data_input["inlay_outer"] = data_input["inlay_outer"].as_open3d

            data_input["fill_undercut"] = event.get("fill_undercut", False)

        stitch_out = run(data_input)

        if event.get("multi_restoration"):
            stitch_out[0].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            stitch_out[0].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

            stitch_out[1].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            stitch_out[1].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

        stitch_json = {
            "crown": compress_drc(stitch_out[0], [stitch_out[2], stitch_out[3], stitch_out[4]]),
            "inner_dilation": write_mesh_bytes(stitch_out[1]),
            "modal_function_call_id": None,
        }
        print("suncess stitch_edge")

        return {"Msg": {"data": stitch_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {
            "error": traceback.format_exc(),
            "modal_function_call_id": None,
        }
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import json

    with open(
        "test_data/0616/b02330f0-d509-41e6-bc95-38f058478d96/post_961b8aa6-46ea-4766-b91c-340f1fdad6c8/output.json"
    ) as f:
        event = json.load(f)
    out = handler(event, None)
    with open('stitch.json', 'w') as f:
        json.dump(out, f)
