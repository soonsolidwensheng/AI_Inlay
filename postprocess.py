import traceback

import trimesh
import numpy as np
from inlay_cpu import InlayGeneration
from utils import read_mesh_bytes, write_mesh_bytes, compress_drc, o3d2tri, get_biggest_mesh


def run(data):
    IG = InlayGeneration(
        tid=data.get("tid"),
        prep_tooth=data.get("prep_tooth"),
        inlay_inner=data.get("inlay_inner"),
        upper_scan=data.get("upper_scan"),
        lower_scan=data.get("lower_scan"),
        # adjacent_teeth=[x for x in data.get("adjacent_teeth")],
        standard=data.get("standard"),
        fill_undercut=data.get("fill_undercut", False),
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
        data_input["inlay_inner"] = get_biggest_mesh(data_input["inlay_inner"]).as_open3d
        data_input["upper_scan"] = read_mesh_bytes(event.get("mesh_upper"))
        data_input["lower_scan"] = read_mesh_bytes(event.get("mesh_lower"))
        # data_input["adjacent_teeth"] = [
        #     read_mesh_bytes(event.get("mesh1")),
        #     read_mesh_bytes(event.get("mesh2")),
        # ]
        data_input["standard"] = read_mesh_bytes(event.get("stdcrown"))
        data_input["tid"] = int(event.get("beiya_id"))
        data_input["paras"] = event.get("paras")
        if event.get("multi_restoration"):
            event["rot_matrix"] = event["rot_matrix"][0]
            event["rot_matrix"][0][-1][-1] = 1
            event["rot_matrix"][1][-1][-1] = 1

            data_input["prep_tooth"] = o3d2tri(data_input["prep_tooth"])
            data_input["prep_tooth"].apply_transform(event["rot_matrix"][0])
            data_input["prep_tooth"].apply_transform(event["rot_matrix"][1])
            data_input["prep_tooth"] = data_input["prep_tooth"].as_open3d

            data_input["inlay_inner"] = o3d2tri(data_input["inlay_inner"])
            data_input["inlay_inner"].apply_transform(event["rot_matrix"][0])
            data_input["inlay_inner"].apply_transform(event["rot_matrix"][1])
            data_input["inlay_inner"] = data_input["inlay_inner"].as_open3d

            data_input["upper_scan"] = o3d2tri(data_input["upper_scan"])
            data_input["upper_scan"].apply_transform(event["rot_matrix"][0])
            data_input["upper_scan"].apply_transform(event["rot_matrix"][1])
            data_input["upper_scan"] = data_input["upper_scan"].as_open3d

            data_input["lower_scan"] = o3d2tri(data_input["lower_scan"])
            data_input["lower_scan"].apply_transform(event["rot_matrix"][0])
            data_input["lower_scan"].apply_transform(event["rot_matrix"][1])
            data_input["lower_scan"] = data_input["lower_scan"].as_open3d

            data_input["standard"] = o3d2tri(data_input["standard"])
            data_input["standard"].apply_transform(event["rot_matrix"][0])
            data_input["standard"].apply_transform(event["rot_matrix"][1])
            data_input["standard"] = data_input["standard"].as_open3d

            data_input["fill_undercut"] = event.get("fill_undercut", False)

            # data_input["adjacent_teeth"][0] = o3d2tri(data_input["adjacent_teeth"][0])
            # data_input["adjacent_teeth"][0].apply_transform(event["rot_matrix"][0])
            # data_input["adjacent_teeth"][0].apply_transform(event["rot_matrix"][1])
            # data_input["adjacent_teeth"][0] = data_input["adjacent_teeth"][0].as_open3d

            # data_input["adjacent_teeth"][1] = o3d2tri(data_input["adjacent_teeth"][1])
            # data_input["adjacent_teeth"][1].apply_transform(event["rot_matrix"][0])
            # data_input["adjacent_teeth"][1].apply_transform(event["rot_matrix"][1])
            # data_input["adjacent_teeth"][1] = data_input["adjacent_teeth"][1].as_open3d

        po_out = run(data_input)

        if event.get("multi_restoration"):
            po_out[0].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            po_out[0].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

            po_out[1].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            po_out[1].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

            po_out[2].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            po_out[2].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

            po_out[3].apply_transform(np.linalg.pinv(event["rot_matrix"][1]))
            po_out[3].apply_transform(np.linalg.pinv(event["rot_matrix"][0]))

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

    # with open("test_data/muti_inlay/0901/inlay_post.json") as f:
    #     data = json.load(f)
    
    # with open("configs.yaml", "r") as file:
    #     config = yaml.safe_load(file)

    # # 修改参数
    # config["isSave"] = True
    # config["savePath"] = "test_data/muti_inlay/0901/result_0825"
    # # 保存修改后的 YAML 文件
    # with open("configs.yaml", "w") as file:
    #     yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    # # if os.path.exists(f"./result/0616/{case}"):
    # #     continue

    # out = handler(data, None)
    # with open("test_data/muti_inlay/0901/post.json", "w") as f:
    #     json.dump(out, f)


    # path = 'test_data/jira/1042'
    # path = "test_data/test_bio"
    # cases = os.listdir(f"{path}")
    # # 3c52791e-c482-458d-9aba-89d6a0a356c3
    # # 04aed962-8013-42e2-a42f-43527810086b
    # # 4d01a81b-c52c-4818-b3e9-11f42953e2a4  磨损
    # # 5c3f7f20-a7c0-407c-bee2-9197d644e545  厚度
    # case_fail = ['a865f1ce-c0eb-4fc4-8b0b-8a220d9644d4']

    # for case in cases:
    #     # case = "1f32fa77-4831-44d4-a0a5-ce684bbf33ba"
    #     # case = "3a2c8200-e010-4278-8734-b02cfa5f8826"
    #     try:
    #         if case in case_fail:
    #             continue
    #         if os.path.exists(f"{path}/{case}/result"):
    #             continue
    #         # f = [x for x in os.listdir(f'{path}/{case}/result') if 'prep_scan' in x][0]
    #         # r = datetime.fromtimestamp(os.path.getmtime(f'{path}/{case}/result/{f}'))
    #         # if r.hour >= 9 and r.day == 30:
    #         #     continue
    #         # gpu_file = [x for x in os.listdir(f"{path}/{case}") if "gpu" in x][0]
    #         # post_file = [x for x in os.listdir(f"{path}/{case}") if "post" in x][0]
    #         with open(f"{path}/{case}/output.json") as f:
    #             data = json.load(f)["cpu_process_info"]
    #         # data = {}
    #         with open(f"{path}/{case}/input.json") as f:
    #             data_ = json.load(f)

    #         # with open(f"{path}/{case}/gpu_result.json") as f:
    #         #     data = json.load(f)
    #         # for key in data['cpu_process_info']:
    #         #     data[key] = data['cpu_process_info'][key]
    #         for key in data_:
    #             data[key] = data_[key]
    #         # data['test_boundary'] = trimesh.load(f"{path}/{case}/stdcrown_morphing_2.stl")
    #         standard = trimesh.load(f"{path}/{case}/bio_mirror.stl")
    #         standard = trimesh.Trimesh.simplify_quadric_decimation(standard, 20000)
    #         data["stdcrown"] = write_mesh_bytes(standard)
    #         # 读取 YAML 文件
    #         with open("configs.yaml", "r") as file:
    #             config = yaml.safe_load(file)

    #         # 修改参数
    #         # config["savePath"] = f"./result/test{i + 110}"
    #         config["isSave"] = True
    #         config["savePath"] = f"{path}/{case}/result"

    #         # 保存修改后的 YAML 文件
    #         with open("configs.yaml", "w") as file:
    #             yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    #         # if os.path.exists(f"./result/0616/{case}"):
    #         #     continue

    #         out = handler(data, None)
    #         with open("post.json", "w") as f:
    #             json.dump(out, f)
    #         # break
    #     except:
    #         print(case)
            # break
    # import pypruners

    # def pruner_fun(mesh: trimesh.Trimesh, interest_verts: np.ndarray, max_distance=1.0, min_distance=-2.5, max_angle=1.5) -> trimesh.Trimesh:
    #     """
    #     通过一组点对网格进行修剪。
    #     参数:
    #     mesh (trimesh.Trimesh): 输入的网格。
    #     interest_verts (numpy.ndarray): 用于修剪的点。
    #     max_distance (float): 修剪的最大距离。
    #     min_distance (float): 修剪的最小距离。
    #     max_angle (float): 修剪的最大角度。
    #     返回:
    #     mesh (trimesh.Trimesh): 修剪后的网格。
    #     """
    #     interest_verts = mesh.nearest.vertex(interest_verts)[1].reshape(-1, 1)
    #     mesh_out = pypruners.pruner(
    #         mesh.vertices, mesh.faces, interest_verts, max_distance, min_distance, max_angle
    #     )
    #     return mesh_out

    # path = r'test_data/muti_inlay/test_data'
    # cases = os.listdir(path)
    # finished_case = [
    #     '2024-08-29_00005-015',  # 嵌体和邻牙？
    #     '06188_20250223_1449_郝伟',  # 边缘相交
    #     '1d78-2542',    # 3号牙邻接
    #     '2023-09-23_98777-010',  # 咬合平面？
    #     '06188_20241028_2018_吕沁柯',  # 嵌体和邻牙
    #     '4ae3-5332',
    #     '06188_20250224_1820_刘芯',
    #     '3240',  # 未接触测试通过
    #     '06188_20240906_2019_胡雨杰',
    #     '2025-03-13_00003-010',  # 边缘相交
    #     '06188_20241106_1620_张淑仪',   # 多个相邻
    #     '2025-03-04_00001-016',     # 未接触测试通过  1个牙冠两个嵌体

    # ]
    # error_case = [
    #     '06188_20250309_1631_陈美梅',  # 传入的模型有自相交  
    #     '1281',  # pruner失败
    # ]
    # for case in cases:
    #     case = '2025-03-07_00003-005'
    #     if case in finished_case:
    #         continue
    #     if case in error_case:
    #         continue
    #     with open(f"{path}/{case}/response_test_new.json", "r") as f:
    #         event_gpu = json.load(f)["Msg"]["data"]

    #     with open(f"{path}/{case}/response_post.json", "r") as f:
    #         event_gpu_post = json.load(f)["Msg"]["data"]
    #     inlay_id = [x for x in event_gpu['inlay_res'].keys()]
    #     for i in inlay_id:
    #         print(case, '---', i)
    #         event = {}
    #         event['mesh_beiya'] = event_gpu['inlay_res'][i]['prep_b']
    #         event['prep_q'] = event_gpu['inlay_res'][i]['prep_q']
    #         event['mesh_upper'] = event_gpu['mesh_upper']
    #         event['mesh_lower'] = event_gpu['mesh_lower']
    #         if i[0] in ['3', '4']:
    #             mesh_jaw = o3d2tri(read_mesh_bytes(event['mesh_lower']))
    #         else:
    #             mesh_jaw = o3d2tri(read_mesh_bytes(event['mesh_upper']))
    #         prep_q = read_mesh_bytes(event['prep_q'])
    #         prep_q = o3d2tri(prep_q)
    #         prep_q = pruner_fun(mesh_jaw, prep_q.vertices)
    #         if len(prep_q.vertices):
    #             event['prep_q'] = write_mesh_bytes(prep_q)
    #         else:
    #             print('pruner filed')
    #         # event['prep_q'] = write_mesh_bytes(trimesh.load("test_data/muti_inlay/case2/prep_q.stl"))
    #         event['mesh1'] = event_gpu['inlay_res'][i]['closer']
    #         event['mesh2'] = event_gpu['inlay_res'][i]['further']
    #         event['stdcrown'] = event_gpu_post[i]['stdcrown']
    #         # event['stdcrown'] = write_mesh_bytes(trimesh.load("test_data/muti_inlay/case2/stdcrown.stl"))
    #         event['beiya_id'] = i
    #         event['multi_restoration'] = True
    #         event['rot_matrix'] = event_gpu['inlay_res'][i]['rot_matrix']
    #         with open("configs.yaml", "r") as file:
    #             config = yaml.safe_load(file)
    #         config["isSave"] = True
    #         config["savePath"] = f"{path}/{case}/{i}"
    #         with open("configs.yaml", "w") as file:
    #             yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    #         out = handler(event, '')
    #         os.makedirs(f"{path}/{case}/post_out", exist_ok=True)
    #         with open(f"{path}/{case}/post_out/post_{i}.json", "w") as f:
    #             f.write(json.dumps(out["Msg"]["data"]))
    #         print
    #     break
    path = r'test_data/c2'
    with open(f"{path}/inlay_post.json", "r") as f:
        event = json.load(f)
    # event['stdcrown'] = write_mesh_bytes(trimesh.load(f'{path}/std_out.stl'))
    with open("configs.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["isSave"] = True
    config["savePath"] = path
    with open("configs.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    out = handler(event, '')


    # path = r'test_data/muti_inlay/prod/20250916-inlay_onlay'
    # cases = os.listdir(path)
    # for case in cases:
    #     case = 'AAAS-6M62'
    #     print(case)
    #     # if os.path.exists(f"{path}/{case}/result_undercut1"):
    #     #     continue
    #     json_path = f"{path}/{case}/inlay_post.json"
    #     with open(json_path, "r") as f:
    #         event = json.load(f)
    #     with open("configs.yaml", "r") as f:
    #         config = yaml.safe_load(f)
    #     config["isSave"] = True
    #     config["savePath"] = f"{path}/{case}/result_undercut1"
    #     with open("configs.yaml", "w") as file:
    #         yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    #     out = handler(event, '')
    #     break
