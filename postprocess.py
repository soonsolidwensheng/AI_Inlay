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

    with open("test_data/studio/AAAD-KSLK/output.json") as f:
        data = json.load(f)["cpu_process_info"]
    
    with open("test_data/studio/AAAD-KSLK/prep_q.json") as f:
        data_ = json.load(f)

    for key in data_:
        data[key] = data_[key]
        
    # with open("test_data/pc_test/output.json") as f:
    #     data = json.load(f)
    
    # data["cpu_process_info"]["prep_q"] = data["mesh_prep"]["S"]
    # data = data["cpu_process_info"]
    data["prep_q"] = data["mesh_prep"]["S"]
    mesh = trimesh.load('result/sdudio_good_case/AAAD-KSLK/2_registreation_36.stl')
    data['stdcrown'] = write_mesh_bytes(mesh)
    
    for i in range(1):
        print(i)
        # 读取 YAML 文件
        with open("configs.yaml", "r") as file:
            config = yaml.safe_load(file)

        # 修改参数
        # config["savePath"] = f"./result/test{i + 110}"
        config["isSave"] = True
        config["savePath"] = "./result/test_KSLK"

        # 保存修改后的 YAML 文件
        with open("configs.yaml", "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        handler(data, None)
        
    # import os
    # import open3d as o3d
    # import numpy as np
    # from utils import angle_between_vectors
    # from icp_w import ipc_exec
    # from stdcrown import run as std_run
    # import time
    
    # pass_list = []
    # cases = os.listdir("test_data/studio")
    # cases_bad = ["AAAD-MAXG", "AAAD-MAN4", "AAAD-MAIX", "AAAD-MADS", "AAAD-KSUT", "AAAD-KSGF"]
    # cases = [x for x in cases if x not in cases_bad]
    # for files in cases:
    #     if files in pass_list:
    #         continue
    #     files = "AAAD-KSLK"
    #     print(files)
    #     with open(f"test_data/studio/{files}/output.json") as f:
    #         data = json.load(f)["cpu_process_info"]
    #     data["prep_q"] = write_mesh_bytes(trimesh.load(f"test_data/studio/{files}/prep_q.stl"))
    #     pcd = o3d.io.read_point_cloud(f"test_data/studio/{files}/inlayonlay_complete.pcd")
    #     s1 = time.time()
    #     pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=60))
    #     pcd.orient_normals_consistent_tangent_plane(100)
    #     pcd_tri = trimesh.PointCloud(pcd.points)
    #     centroid = pcd_tri.centroid
    #     if angle_between_vectors(np.asarray(pcd.normals)[0], pcd_tri.vertices[0] - centroid) > np.pi / 2:
    #         pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1)
        
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.3]))
    #     s2 = time.time()
    #     print("pcd2mesh", s2 - s1)
    #     # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.5, linear_fit=False,)[0]
        
    #     # partial = trimesh.load(f"test_data/studio/{files}/mesh_partial.stl")
    #     # prod = trimesh.proximity.ProximityQuery(partial)
    #     # _, dis, _ = prod.on_surface(np.asarray(mesh.vertices))
    #     # idx = np.where(dis < 0.05)[0]
    #     # mesh.remove_vertices_by_index(idx)
         
    #     mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
        
    #     # mesh = mesh.split(only_watertight=False)
    #     # mesh = mesh[np.argmax(np.array([x.vertices.shape[0] for x in mesh]))]
    #     std_mesh = std_run({"beiya_id": data.get("beiya_id")})
    #     s3 = time.time()
    #     std_mesh_copy = std_mesh.copy()
    #     std_mesh_copy = std_mesh_copy.simplify_quadric_decimation(face_count=10000)
    #     mesh = mesh.simplify_quadric_decimation(face_count=6000)
    #     # mesh.export(f"./result/sdudio/{files}/mesh_complete.stl")
    #     mat = ipc_exec(std_mesh_copy, mesh, 30)
    #     std_mesh = std_mesh.apply_transform(mat)
    #     print("ipc_exec", time.time() - s3)
    #     data['stdcrown'] = write_mesh_bytes(std_mesh)
        
    #     for i in range(1):
    #         # 读取 YAML 文件
    #         with open("configs.yaml", "r") as file:
    #             config = yaml.safe_load(file)

    #         # 修改参数
    #         # config["savePath"] = f"./result/test{i + 110}"
    #         config["isSave"] = True
    #         config["savePath"] = f"./result/sdudio/{files}"

    #         # 保存修改后的 YAML 文件
    #         with open("configs.yaml", "w") as file:
    #             yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    #         handler(data, None)
    #     break