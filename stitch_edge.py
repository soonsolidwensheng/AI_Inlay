import base64
import traceback

import DracoPy
import numpy as np
import trimesh

from inlay_cpu import InlayGeneration, MeshRegistration


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
    )
    mesh_registration = MeshRegistration(IG.configs)
    mesh_registration.get_cement_gap(IG.o3d2tri(IG.inlay_inner))
    # boundarySet, _ = mesh_registration.getBoundaryPoints(mesh_registration.inner_dilation.as_open3d)
    # IG.lib_tooth = mesh_registration.doSubMeshTps(boundarySet, IG.lib_tooth, mesh_registration.lastSubmehs_tps_dis)
    IG.inner_dilation = mesh_registration.inner_dilation
    IG.inlay_outer = IG.lib_tooth
    IG.stitch()

    return IG.o3d2tri(IG.get_stitched_inlay()), IG.inner_dilation


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
        stitch_out = run(data_input)

        stitch_json = {
            "crown": write_mesh_bytes(stitch_out[0]),
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
    with open('test_data/41b82e63-ac82-4522-b8b5-6012e35445df/post_c005ab3b-0c2a-43d1-81ef-2ab5bf66b884/output.json') as f:
        event = json.load(f)
    print(handler(event, None))