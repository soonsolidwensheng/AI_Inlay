import traceback

import numpy as np
import pymeshrepairer
import trimesh

from utils import read_mesh_bytes, write_mesh_bytes


def mesh_repair(vertices, faces):
    # mesh repair
    mesh_r = pymeshrepairer.Mesh()
    mesh_r.read(vertices, faces)

    # min_edge_length (default: 1.0e-4)
    # 用于判定退化边和退化面
    # 如果边长小于此值，则认为是退化边
    # 如果面的边长或高小于此值，则认为是退化面
    min_edge_length = 1.0e-6

    # min_split_edge_length (default: 1.0e-2)
    # 在涉及到拆分边的操作时，假如只会拆分大于等于此值的边
    min_split_edge_length = 1.0e-2

    # max_isolated_pieces_diameter (default: 1.0e-1)
    # 用于删除非水密的小碎片，当非水密的碎片直径小于此值，被定义为需要删除的小碎片
    max_isolated_pieces_diameter = 1.0e6

    # max_iterations (default: 10)
    # 在进行每一步的修复操作时，最大尝试修复操作次数
    max_iterations = 10

    success01, report = pymeshrepairer.repair(
        mesh_r,
        min_edge_length,
        min_split_edge_length,
        max_isolated_pieces_diameter,
        max_iterations,
    )

    # print("First repair")
    # print(success01)
    # print(report)

    # if not success01:
    #     success01, report = pymeshrepairer.repair(mesh_r, min_edge_length, min_split_edge_length, max_isolated_pieces_diameter, max_iterations)
    #     # print("Second repair")
    #     # print(success01)
    #     # print(report)

    your_vertices_data, your_faces_data = mesh_r.write()

    return your_vertices_data, your_faces_data, success01, report


import pyholefiller


def mesh_fillhole(vertices, faces):
    # mesh_fillhole

    mesh_f = pyholefiller.Mesh()
    mesh_f.vertices = vertices
    mesh_f.faces = faces

    # fill mesh holes
    maximum_filling_hole_size = 6.28

    success02 = pyholefiller.fill_hole(mesh_f, maximum_filling_hole_size)

    # check result
    mesh_r = pymeshrepairer.Mesh()
    min_edge_length = 1.0e-6

    mesh_r.read(mesh_f.vertices, mesh_f.faces)
    success01, report01 = pymeshrepairer.analyze(mesh_r, min_edge_length)

    if not success02 or not success01:
        mesh_f.vertices, mesh_f.faces, success01, report01 = mesh_repair(
            mesh_f.vertices, mesh_f.faces
        )

        success02 = pyholefiller.fill_hole(mesh_f, maximum_filling_hole_size)

        # check result
        mesh_r.read(mesh_f.vertices, mesh_f.faces)
        success01, report01 = pymeshrepairer.analyze(mesh_r, min_edge_length)

        if not success02 or not success01:
            mesh_f.vertices, mesh_f.faces, success01, report01 = mesh_repair(
                mesh_f.vertices, mesh_f.faces
            )

    # check result
    mesh_r.read(mesh_f.vertices, mesh_f.faces)
    success01, report01 = pymeshrepairer.analyze(mesh_r, min_edge_length)

    your_vertices_data = mesh_f.vertices
    your_faces_data = mesh_f.faces

    return your_vertices_data, your_faces_data, success01, success02, report01


import pylfda


def mesh_decimate(vertices, faces, desired_vertex_count):
    mesh = pylfda.Mesh()
    mesh.vertices = vertices
    mesh.faces = faces

    decimation_type = pylfda.DecimationType.Vertex
    max_normal_deviation = 10

    success01 = pylfda.decimate_mesh(
        mesh, desired_vertex_count, decimation_type, max_normal_deviation
    )

    your_vertices_data = mesh.vertices
    your_faces_data = mesh.faces

    return your_vertices_data, your_faces_data, success01


def mesh_subdivide(vertices, faces, max_edge_length):
    mesh = pylfda.Mesh()
    mesh.vertices = vertices
    mesh.faces = faces

    # subdivide mesh
    # max_edge_length = 0.8
    success01 = pylfda.subdivide_mesh(mesh, max_edge_length)

    your_vertices_data = mesh.vertices
    your_faces_data = mesh.faces

    return your_vertices_data, your_faces_data, success01


def repaire_mesh(upper_v, upper_f):
    try:
        step01_dv, step01_df, success01, report01 = mesh_repair(upper_v, upper_f)
        if success01 and step01_dv.shape[0] > 50000:
            step02_dv, step02_df, success03 = mesh_decimate(step01_dv, step01_df)
            if success03:
                upper_v, upper_f, success05, report02 = mesh_repair(
                    step02_dv, step02_df
                )
            else:
                upper_v, upper_f = step01_dv, step01_df
        elif not success01:
            upper_v, upper_f = upper_v, upper_f
        elif success01 and step01_dv.shape[0] <= 50000:
            upper_v, upper_f, success05, report02 = mesh_repair(step01_dv, step01_df)
    except:
        upper_v, upper_f = upper_v, upper_f

    return upper_v, upper_f


def run(data):
    mesh_upper = data["upper"]
    mesh_lower = data["lower"]

    if not isinstance(mesh_upper, trimesh.Trimesh):
        mesh_upper = trimesh.Trimesh(
            np.asarray(mesh_upper.vertices), np.asarray(mesh_upper.triangles)
        )
    if not isinstance(mesh_lower, trimesh.Trimesh):
        mesh_lower = trimesh.Trimesh(
            np.asarray(mesh_lower.vertices), np.asarray(mesh_lower.triangles)
        )

    if not mesh_upper.is_watertight and not mesh_lower.is_watertight:
        upper_v, upper_f = mesh_upper.vertices, mesh_upper.faces
        lower_v, lower_f = mesh_lower.vertices, mesh_lower.faces
        upper_v, upper_f = repaire_mesh(upper_v, upper_f)
        lower_v, lower_f = repaire_mesh(lower_v, lower_f)
    else:
        upper_v, upper_f = mesh_upper.vertices, mesh_upper.faces
        lower_v, lower_f = mesh_lower.vertices, mesh_lower.faces

    return trimesh.Trimesh(upper_v, upper_f), trimesh.Trimesh(lower_v, lower_f)


def run_single(data):
    mesh = data["mesh"]
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    if not mesh.is_watertight:
        v, f = mesh.vertices, mesh.faces
        v, f = repaire_mesh(v, f)
    else:
        v, f = mesh.vertices, mesh.faces
    return trimesh.Trimesh(v, f)


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_Repair ..")
        if event.get("job_id"):
            print(f"job_id: {event.get('job_id')}")
        else:
            print("job_id: None")
        if event.get("execution_id"):
            print(f"execution_id: {event.get('execution_id')}")
        else:
            print("execution_id: None")
        data_input = {}
        if event.get("single_mesh"):
            data_input["mesh"] = read_mesh_bytes(event.get("mesh"))
            repaired_out = run_single(data_input)
            repaired_json = {
                "repaired_mesh": repaired_out,
                "modal_function_call_id": None,
            }
        else:
            data_input["upper"] = read_mesh_bytes(event.get("mesh_upper"))
            data_input["lower"] = read_mesh_bytes(event.get("mesh_lower"))
            repaired_out = run(data_input)
            repaired_json = {
                "mesh_upper": write_mesh_bytes(repaired_out[0]),
                "mesh_lower": write_mesh_bytes(repaired_out[1]),
                "modal_function_call_id": None,
            }
        print("suncess mesh repair")
        return {"Msg": {"data": repaired_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {
            "error": traceback.format_exc(),
            "modal_function_call_id": None,
        }
        traceback.print_exc()
        return res


if __name__ == "__main__":
    pass
    import time
    # load mesh data into vertex array and face array

    # mesh_upper = trimesh.Trimesh(upper_v, upper_f)
    # mesh_lower = trimesh.Trimesh(lower_v, lower_f)
    # mesh_upper = trimesh.load_mesh("upper.stl")
    # mesh_lower = trimesh.load_mesh("lower.stl")
    # upper_v, upper_f = mesh_upper.vertices, mesh_upper.faces
    # lower_v, lower_f = mesh_lower.vertices, mesh_lower.faces
    # start_time = time.time()
    # if not mesh_upper.is_watertight and not mesh_lower.is_watertight:

    #     # upper_v1, upper_f1 = mesh_repair(upper_v, upper_f)
    #     step01_dv, step01_df, success01, report01 = mesh_repair(upper_v, upper_f)
    #     if success01 and step01_dv.shape[0] > 50000:
    #         mesh_upper = trimesh.Trimesh(step01_dv, step01_df).simplify_quadric_decimation(100000)
    #         step02_dv, step02_df = mesh_upper.vertices, mesh_upper.faces
    #         success03 = True
    #         # step02_dv, step02_df , success03 = mesh_decimate(step01_dv, step01_df, 10000)
    #         if success03:
    #             upper_v, upper_f, success05, report02 = mesh_repair(step02_dv, step02_df)
    #         else:
    #             upper_v, upper_f = step01_dv, step01_df
    #     elif not success01:
    #         upper_v, upper_f = upper_v, upper_f
    #     elif success01 and step01_dv.shape[0] <= 50000:
    #         upper_v, upper_f, success05, report02 = mesh_repair(step01_dv, step01_df)

    #     step01_dv, step01_df, success01, report01 = mesh_repair(lower_v, lower_f)
    #     if success01 and step01_dv.shape[0] > 50000:
    #         mesh_lower = trimesh.Trimesh(step01_dv, step01_df).simplify_quadric_decimation(100000)
    #         step02_dv, step02_df = mesh_lower.vertices, mesh_lower.faces
    #         success03 = True
    #         # step02_dv, step02_df , success03 = mesh_decimate(step01_dv, step01_df, 100000)
    #         if success03:
    #             lower_v, lower_f, success05, report02 = mesh_repair(step02_dv, step02_df)
    #         else:
    #             lower_v, lower_f = step01_dv, step01_df
    #     elif not success01:
    #         lower_v, lower_f= lower_v, lower_f
    #     elif success01 and step01_dv.shape[0] <= 50000:
    #         lower_v, lower_f, success05, report02 = mesh_repair(step01_dv, step01_df)
    # else:
    #     upper_v, upper_f = upper_v, upper_f
    #     lower_v, lower_f = lower_v, lower_f
    # print(time.time() - start_time)
    # trimesh.Trimesh(upper_v, upper_f).export("upper_repaired.stl")
    # trimesh.Trimesh(lower_v, lower_f).export("lower_repaired.stl")

    mesh = trimesh.load("0.stl")
    v, f = mesh.vertices, mesh.faces
    s = time.time()
    # step02_dv, step02_df , success03 = mesh_decimate(v, f, 10000)
    mesh = mesh.simplify_quadric_decimation(20000)
    print(time.time() - s)
    # trimesh.Trimesh(step02_dv, step02_df).export("00_decimate.stl")
    mesh.export("00_decimate.stl")
