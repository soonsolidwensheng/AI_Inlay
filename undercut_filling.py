import json
import traceback
import base64
import DracoPy
import numpy as np
import trimesh
from crown_cpu import undercut_filling


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
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def write_drc(drc, file):
    with open(file, "wb") as test_file:
        test_file.write(base64.b64decode(drc))


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Crown_CPU_Undercut_Filling ..")
        print("job_id", event.get("job_id"))
        print("event keys", event.keys())
        print("AOI", event.get("AOI"))
        cpu_input_json = {
            "inner": event.get("inner"),
            "AOI_or_UB": event.get("AOI_or_UB"),
            "AOI": event.get("AOI"),
            "prep_extended": event.get("prep_extended"),
        }
        filling_out = undercut_filling(event)
        # filling_out.mesh_beiya.export(os.path.join(context, 'inner.stl'))
        # filling_out.undercut_mesh.export(os.path.join(context, 'vox.stl'))
        if filling_out.AOI_or_UB == 0:
            insert_direction = filling_out.insert_direction.tolist()
        elif filling_out.AOI_or_UB == 1:
            print(len(filling_out.mesh_beiya.vertices))
            mesh_beiya = write_mesh_bytes(filling_out.mesh_beiya)
            insert_direction = filling_out.insert_direction

        if filling_out.AOI_or_UB == 0:
            undercut_json = {
                "AOI": insert_direction,
                "inner": None,
                "cpu_input_json": cpu_input_json,
            }
        elif filling_out.AOI_or_UB == 1:
            undercut_json = {
                "AOI": insert_direction,
                "inner": mesh_beiya,
                "cpu_input_json": cpu_input_json,
            }

        print("suncess undercut")

        return {"Msg": {"data": undercut_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {"Msg": traceback.format_exc(), "Code": 203, "State": "Failure"}
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import time
    import os

    save_dir = r"/home/wanglong/PycharmProjects/lambda_crown/cad_git/download"
    cases = os.listdir(save_dir)
    for case_id in cases:
        # case_id = "6550a184-a765-4a94-9500-ad3936327208"
        print(case_id)
        with open(os.path.join(save_dir, case_id, "gpu-step-2-result.json"), "r") as f:
            event = json.load(f)
        s1 = time.time()
        event["AOI_or_UB"] = 1
        event["inner"] = event["prep"]
        out = handler(event, os.path.join(save_dir, case_id))
        s2 = time.time()
        print(s2 - s1)
        event['inner'] = out["Msg"]["data"]['inner']
        with open(os.path.join(save_dir, case_id, "undercut_filling_out.json"), "w") as f:
            f.write(json.dumps(event))
        # break
    
    # save_dir = 'test_data_'
    # case_id = '3c8a'
    
    # with open(os.path.join(save_dir, case_id, "undercut_input.json"), "r") as f:
    #     event = json.load(f)
    
    # # with open(os.path.join(save_dir, case_id, "gpu-step-2-result.json"), "r") as f:
    # #     event['prep_extended'] = json.load(f)["prep_extended"]
    # # event["AOI_or_UB"] = 1
    # # with open(os.path.join(save_dir, case_id, "crown_std.json"), "r") as f:
    # #     event_ = json.load(f)
    
    # # with open(os.path.join(save_dir, case_id, "std.json"), "r") as f:
    # #     event_s = json.load(f)
    # # event['prep_extended'] = event_['prep_extended']
    # # event_['cpu_std_json']['cpu_colors_info'] = event_s['cpu_std_json']['cpu_colors_info']
    # # event['cpu_std_json'] = event_['cpu_std_json']
    
    # out = handler(event, os.path.join(save_dir, case_id))
    
    # print