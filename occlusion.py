import traceback

from inlay_cpu import InlayGeneration
from utils import compress_drc, read_mesh_bytes, write_mesh_bytes


def run(data):
    IG = InlayGeneration(
        tid=data.get("tid"),
        prep_tooth=data.get("prep_tooth"),
        inlay_inner=data.get("inner_dilation"),
        upper_scan=data.get("upper_scan"),
        lower_scan=data.get("lower_scan"),
        adjacent_teeth=[x for x in data.get("adjacent_teeth")],
        standard=data.get("inlay_outer"),
    )
    IG.run_occlu()

    return {
        "stitched_inlay": IG.o3d2tri(IG.stitched_inlay),
        "inlay_outer": IG.o3d2tri(IG.inlay_outer),
        "fixed_stitched_inlay": IG.o3d2tri(IG.fixed_stitched_inlay),
        "fixed_inlay_outer": IG.o3d2tri(IG.fixed_inlay_outer),
        "thickness_points": IG.thickness_points_id,
    }


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_Occlu ..")
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
        data_input["upper_scan"] = read_mesh_bytes(event.get("mesh_upper"))
        data_input["lower_scan"] = read_mesh_bytes(event.get("mesh_lower"))
        data_input["adjacent_teeth"] = [
            read_mesh_bytes(event.get("mesh1")),
            read_mesh_bytes(event.get("mesh2")),
        ]
        data_input["inlay_outer"] = read_mesh_bytes(event.get("inlay_outer"))
        data_input["tid"] = int(event.get("beiya_id"))
        occlu_out = run(data_input)
        occlu_json = {
            "crown": write_mesh_bytes(occlu_out["stitched_inlay"]),
            "inlay_outer": write_mesh_bytes(occlu_out["inlay_outer"]),
            "fixed_crown": compress_drc(
                occlu_out["fixed_stitched_inlay"], occlu_out["thickness_points"]
            ),
            "fixed_inlay_outer": write_mesh_bytes(occlu_out["fixed_inlay_outer"]),
            "modal_function_call_id": None,
        }
        print("suncess occlusion")

        return {"Msg": {"data": occlu_json}, "Code": 200, "State": "Success"}
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

    with open("test_data/2302911b-ad65-4681-9b1f-085ac15590ab/gpu_7377fc8c-4c35-4d45-969d-50f521dd6493/output.json") as f:
        data = json.load(f)["cpu_process_info"]
    
    with open("test_data/2302911b-ad65-4681-9b1f-085ac15590ab/post_0b2bcfb3-208b-48d5-8f64-3dc2ed484145/output.json") as f:
        data_ = json.load(f)

    with open("test_data/2302911b-ad65-4681-9b1f-085ac15590ab/occ_0c6194f4-adc1-4772-82e7-afeebc83533d/input.json") as f:
        data__ = json.load(f)
        
    for key in data_:
        data[key] = data_[key]
    
    for key in data__:
        data[key] = data__[key] 
    # with open("test_data/ali/post_error/1924366727035092993/post_1924367556224086016/postCrownInput.json") as f:
    #     data = json.load(f)
    
    # mesh = trimesh.load('test_data/e6a9294b-7324-46a7-87b9-b37196907864/0_lib_tooth_16.ply')
    # data['stdcrown'] = write_mesh_bytes(mesh)
    
    for i in range(1):
        print(i)
        # 读取 YAML 文件
        with open("configs.yaml", "r") as file:
            config = yaml.safe_load(file)

        # 修改参数
        # config["savePath"] = f"./result/test{i + 110}"
        config["isSave"] = True
        config["savePath"] = "./result/test_2302911b"

        # 保存修改后的 YAML 文件
        with open("configs.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        handler(data, None)