import traceback

import trimesh

from inlay_cpu import InlayGeneration
from utils import read_mesh_bytes, write_mesh_bytes


def run(data):
    IG = InlayGeneration(
        tid=data.get("tid"),
        prep_tooth=data.get("prep_tooth"),
        inlay_inner=data.get("inlay_inner"),
        upper_scan=data.get("upper_scan"),
        lower_scan=data.get("lower_scan"),
        adjacent_teeth=[x for x in data.get("adjacent_teeth")],
        standard=data.get("standard"),
        paras=data.get("paras"),
    )
    IG.run()

    stitched_inlay, inlay_outer, inner_dilation = (
        IG.get_stitched_inlay(),
        IG.get_inlay_outer(),
        IG.get_inlay_inner(),
    )

    if not isinstance(stitched_inlay, trimesh.Trimesh):
        stitched_inlay = IG.o3d2tri(stitched_inlay)
    if not isinstance(inlay_outer, trimesh.Trimesh):
        inlay_outer = IG.o3d2tri(inlay_outer)
    if not isinstance(inner_dilation, trimesh.Trimesh):
        inner_dilation = IG.o3d2tri(inner_dilation)

    return stitched_inlay, inlay_outer, inner_dilation


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_Post ..")
        if event.get("job_id"):
            print(f"job_id: {event.get('job_id')}")
        else:
            print("job_id: None")
        if event.get("execution_id"):
            print(f"execution_id: {event.get('execution_id')}")
        else:
            print("execution_id: None")
        data_input = {}
        data_input["prep_tooth"] = read_mesh_bytes(event.get("mesh_beiya"))
        data_input["inlay_inner"] = read_mesh_bytes(event.get("prep_q"))
        data_input["upper_scan"] = read_mesh_bytes(event.get("mesh_upper"))
        data_input["lower_scan"] = read_mesh_bytes(event.get("mesh_lower"))
        data_input["adjacent_teeth"] = [
            read_mesh_bytes(event.get("mesh1")),
            read_mesh_bytes(event.get("mesh2")),
        ]
        data_input["standard"] = read_mesh_bytes(event.get("stdcrown"))
        data_input["tid"] = int(event.get("beiya_id"))
        data_input["paras"] = event.get("paras")
        po_out = run(data_input)

        post_json = {
            "crown": write_mesh_bytes(po_out[0]),
            "inlay_outer": write_mesh_bytes(po_out[1]),
            "inner_dilation": write_mesh_bytes(po_out[2]),
            "modal_function_call_id": None,
        }
        print("suncess postprocess")

        return {"Msg": {"data": post_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {
            "error": traceback.format_exc(),
            "modal_function_call_id": None,
        }
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import json
    import yaml

    with open("test_data/e36b21e2-d3a4-4eae-a573-3a20c0d63f7f/gpu_f7ee2f13-1523-412a-bdea-8ab4908a29cc/output.json") as f:
        data = json.load(f)["cpu_process_info"]
    
    with open("test_data/e36b21e2-d3a4-4eae-a573-3a20c0d63f7f/post_d9d42b7f-b6e2-4787-a20d-32969a443c11/input.json") as f:
        data_ = json.load(f)

    for key in data_:
        data[key] = data_[key]
    
    for i in range(1):
        print(i)
        # 读取 YAML 文件
        with open("configs.yaml", "r") as file:
            config = yaml.safe_load(file)

        # 修改参数
        # config["savePath"] = f"./result/test{i + 117}"
        config["savePath"] = "./result/test_e36b21e2"

        # 保存修改后的 YAML 文件
        with open("configs.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        handler(data, None)