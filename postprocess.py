import traceback

import trimesh

from inlay_cpu import InlayGeneration
from utils import read_mesh_bytes, write_mesh_bytes, compress_drc


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
    )


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
            "crown": compress_drc(po_out[0], [po_out[4], po_out[5], po_out[6]]),
            "inlay_outer": write_mesh_bytes(po_out[1]),
            "inner_dilation": write_mesh_bytes(po_out[2]),
            "thickness_shell": write_mesh_bytes(po_out[3]),
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
    import os
    from datetime import datetime

    # path = 'test_data/jira/1042'
    path = "test_data/no_adj"
    cases = os.listdir(f"{path}")
    # 3c52791e-c482-458d-9aba-89d6a0a356c3
    # 04aed962-8013-42e2-a42f-43527810086b
    # 4d01a81b-c52c-4818-b3e9-11f42953e2a4  磨损
    # 5c3f7f20-a7c0-407c-bee2-9197d644e545  厚度

    for case in cases:
        # case = "1f32fa77-4831-44d4-a0a5-ce684bbf33ba"
        # case = "eeff0902-d1e0-47f0-93f3-44ce659baa7b"
        try:
            # f = [x for x in os.listdir(f'{path}/{case}/result') if 'prep_scan' in x][0]
            # r = datetime.fromtimestamp(os.path.getmtime(f'{path}/{case}/result/{f}'))
            # if r.hour >= 9 and r.day == 30:
            #     continue
            gpu_file = [x for x in os.listdir(f"{path}/{case}") if "gpu" in x][0]
            post_file = [x for x in os.listdir(f"{path}/{case}") if "post" in x][0]
            with open(f"{path}/{case}/{gpu_file}/output.json") as f:
                data = json.load(f)["cpu_process_info"]

            with open(f"{path}/{case}/{post_file}/input.json") as f:
                data_ = json.load(f)

            for key in data_:
                data[key] = data_[key]

            # standard = trimesh.load(f"test_data/0617/{case}/std_mesh.stl")
            # data["stdcrown"] = write_mesh_bytes(standard)
            # 读取 YAML 文件
            with open("configs.yaml", "r") as file:
                config = yaml.safe_load(file)

            # 修改参数
            # config["savePath"] = f"./result/test{i + 110}"
            config["isSave"] = True
            config["savePath"] = f"{path}/{case}/result"

            # 保存修改后的 YAML 文件
            with open("configs.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

            # if os.path.exists(f"./result/0616/{case}"):
            #     continue

            out = handler(data, None)
            with open('post.json', 'w') as f:
                json.dump(out, f)
            break
        except:
            print(case)
            break
