import base64
import json
import traceback
import DracoPy
import MQCompressPy
import numpy as np
import trimesh

from crown_cpu import stdcrown


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


def write_mesh_bytes(mesh, colors=None, preserve_order=False):
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


def compress_drc(mesh, points_id=[]):
    vert_flags = np.zeros(len(mesh.vertices), dtype=np.uint8)
    vert_flags[points_id] = 1
    # trimesh.PointCloud(mesh.vertices[points_id]).export('p.ply')
    in_mesh = MQCompressPy.MQC_Mesh()
    in_mesh.verts = MQCompressPy.VerticeArray(mesh.vertices)
    in_mesh.faces = MQCompressPy.FaceArray(mesh.faces)
    in_vert_flags = MQCompressPy.VerticeFlag_UINT8(
        np.array(vert_flags).astype(np.uint8)
    )
    compressed_data, error_code = MQCompressPy.compressMesh_UINT8(
        in_mesh, in_vert_flags
    )
    # with open('mesh.drc', 'wb') as f:
    #     f.write(compressed_data)
    if error_code == 0:
        b64_bytes = base64.b64encode(compressed_data)
        b64_str = b64_bytes.decode("utf-8")
        return b64_str
    else:
        assert "drc compress error"


def matrix2matrix(crown_rot_matirx):
    ai_matrix = np.eye(4)  # 创建一个单位矩阵作为变换矩阵的初始值
    ai_matrix[:3, :3] = crown_rot_matirx  # 复制旋转矩阵的前三列到变换矩阵的前三列
    ai_matrix[:, 3] = [0, 0, 0, 1]
    return ai_matrix

def file_name2template_name(file_name):
    s = file_name.split('_')[0]
    f2t = {
        'CSv1': 'Cyber Standard v1',
        'GSv1': 'Generic Standard v1',
        'Mv1': 'Mature v1',
        'rev1': 'st_tooth',
        'SSv1': 'Soft Standard v1',
        'Yv1': 'Youth v1'
        }
    return f2t[s]

def handler(event, context):
    print("receive case")
    try:
        input_info = event
        try:
            input_info["ai_matrix"] = matrix2matrix(input_info["crown_rot_matirx"])
            new_transform = input_info["new_transform_list"]
            mesh_beiya = input_info["mesh_beiya"]
            mesh_beiya = read_mesh_bytes(mesh_beiya)
            mesh_beiya.apply_transform(np.linalg.pinv(input_info["ai_matrix"]))
            mesh_beiya.apply_transform(np.linalg.pinv(new_transform[2]))
            mesh_beiya.apply_transform(np.linalg.pinv(new_transform[0]))
            mesh_beiya.apply_transform(np.linalg.pinv(new_transform[1]))
            mesh_beiya = write_mesh_bytes(mesh_beiya)
            input_info["mesh_beiya"] = mesh_beiya
            mesh_upper = input_info["mesh_upper"]
            mesh_upper = read_mesh_bytes(mesh_upper)
            mesh_upper.apply_transform(np.linalg.pinv(input_info["ai_matrix"]))
            mesh_upper.apply_transform(np.linalg.pinv(new_transform[2]))
            mesh_upper.apply_transform(np.linalg.pinv(new_transform[0]))
            mesh_upper.apply_transform(np.linalg.pinv(new_transform[1]))
            mesh_upper = write_mesh_bytes(mesh_upper)
            input_info["mesh_upper"] = mesh_upper
            mesh_lower = input_info["mesh_lower"]
            mesh_lower = read_mesh_bytes(mesh_lower)
            mesh_lower.apply_transform(np.linalg.pinv(input_info["ai_matrix"]))
            mesh_lower.apply_transform(np.linalg.pinv(new_transform[2]))
            mesh_lower.apply_transform(np.linalg.pinv(new_transform[0]))
            mesh_lower.apply_transform(np.linalg.pinv(new_transform[1]))
            mesh_lower = write_mesh_bytes(mesh_lower)
            input_info["mesh_lower"] = mesh_lower
            mesh1 = input_info["mesh1"]
            mesh1 = read_mesh_bytes(mesh1)
            mesh1.apply_transform(np.linalg.pinv(input_info["ai_matrix"]))
            mesh1.apply_transform(np.linalg.pinv(new_transform[2]))
            mesh1.apply_transform(np.linalg.pinv(new_transform[0]))
            mesh1.apply_transform(np.linalg.pinv(new_transform[1]))
            mesh1 = write_mesh_bytes(mesh1)
            input_info["mesh1"] = mesh1
            mesh2 = input_info["mesh2"]
            mesh2 = read_mesh_bytes(mesh2)
            mesh2.apply_transform(np.linalg.pinv(input_info["ai_matrix"]))
            mesh2.apply_transform(np.linalg.pinv(new_transform[2]))
            mesh2.apply_transform(np.linalg.pinv(new_transform[0]))
            mesh2.apply_transform(np.linalg.pinv(new_transform[1]))
            mesh2 = write_mesh_bytes(mesh2)
            input_info["mesh2"] = mesh2
        except Exception:
            pass
        print("pred_filestem_name", input_info.get("pred_filestem_name"))
        if input_info.get("pred_filestem_name"):
            input_info["template_name"] = file_name2template_name(input_info.get("pred_filestem_name"))
        print("template_name", input_info.get("template_name"))
        print("job_id", input_info.get("job_id"))
        print("start AI_Crown_CPU_Std ..")
        std_out = stdcrown(input_info)
        cpu_points_info = {}
        cpu_colors_info = {}
        colors = np.zeros_like(std_out.mesh.vertices, dtype=np.uint8)
        points_keys = [
            x for x in vars(std_out.mesh) if x not in vars(trimesh.Trimesh())
        ][2:]
        color_n = 0
        for key in points_keys:
            points = getattr(std_out.mesh, key, None)
            if points:
                color_list = []
                for p_n, p_idx in enumerate(points.idx):
                    color_r, color_g, color_b = colors[p_idx]
                    color_str = (
                        bin(color_r)[2:].zfill(8)
                        + bin(color_g)[2:].zfill(8)
                        + bin(color_b)[2:].zfill(8)
                    )
                    color_str = color_str[:color_n] + "1" + color_str[color_n + 1 :]
                    if key in [
                        "ad_points",
                    ]:
                        color_str = color_str[:-8] + bin(p_n)[2:].zfill(4) + color_str[-4:]
                    if key in [
                        "cross_points",
                    ]:
                        color_str = color_str[:-4] + bin(p_n)[2:].zfill(4)
                    color_r = int(color_str[:8], 2)
                    color_g = int(color_str[8:16], 2)
                    color_b = int(color_str[16:], 2)
                    colors[p_idx] = [color_r, color_g, color_b]
                    if [color_r, color_g, color_b] not in color_list:
                        color_list.append([color_r, color_g, color_b])
                cpu_colors_info[key] = color_list
                color_n += 1

                cpu_points_info[key] = points.pt.tolist()
                # cpu_points_info[key] = points.idx.tolist()
        cpu_points_info_backup = {}
        cpu_colors_info_backup = {}
        colors_backup = np.zeros_like(std_out.mesh_backup.vertices, dtype=np.uint8)
        points_keys = [
            x for x in vars(std_out.mesh) if x not in vars(trimesh.Trimesh())
        ][2:]
        # color = 100
        color_n = 0
        for key in points_keys:
            points = getattr(std_out.mesh, key, None)
            if points:
                color_list = []
                for p_n, p_idx in enumerate(points.idx):
                    color_r, color_g, color_b = colors_backup[p_idx]
                    color_str = (
                        bin(color_r)[2:].zfill(8)
                        + bin(color_g)[2:].zfill(8)
                        + bin(color_b)[2:].zfill(8)
                    )
                    color_str = color_str[:color_n] + "1" + color_str[color_n + 1 :]
                    if key in [
                        "ad_points",
                    ]:
                        color_str = color_str[:-8] + bin(p_n)[2:].zfill(4) + color_str[-4:]
                    if key in [
                        "cross_points",
                    ]:
                        color_str = color_str[:-4] + bin(p_n)[2:].zfill(4)
                    color_r = int(color_str[:8], 2)
                    color_g = int(color_str[8:16], 2)
                    color_b = int(color_str[16:], 2)
                    colors_backup[p_idx] = [color_r, color_g, color_b]
                    if [color_r, color_g, color_b] not in color_list:
                        color_list.append([color_r, color_g, color_b])
                cpu_colors_info_backup[key] = color_list
                color_n += 1

                cpu_points_info_backup[key] = points.pt.tolist()
                # cpu_points_info_backup[key] = points.idx.tolist()
        points_oppo_id = std_out.points_oppo_id
        # points_drc_id = list(range(len(std_out.mesh.vertices)))
        # standard = compress_drc(std_out.mesh, points_drc_id)
        # mesh_backup = compress_drc(std_out.mesh_backup, points_drc_id)
        standard = write_mesh_bytes(std_out.mesh, colors)
        mesh_backup = write_mesh_bytes(std_out.mesh_backup, colors_backup)
        closer = write_mesh_bytes(std_out.mesh1)
        further = write_mesh_bytes(std_out.mesh2)
        mesh_beiya = write_mesh_bytes(std_out.mesh_beiya)
        mesh_upper = write_mesh_bytes(std_out.mesh_upper)
        mesh_lower = write_mesh_bytes(std_out.mesh_lower)
        mesh_jaw = write_mesh_bytes(std_out.mesh_jaw)
        mesh_oppo = write_mesh_bytes(std_out.mesh_oppo)
        template_name = std_out.template_name
        # std_out.mesh_lower.export(os.path.join(context, 'lower.stl'))
        # std_out.mesh_upper.export(os.path.join(context, 'upper.stl'))
        # std_out.mesh.export(os.path.join(context, 'standard.stl'))
        # std_out.mesh_lower.export('lower.stl')
        # std_out.mesh_upper.export('upper.stl')
        # std_out.mesh.export('standard1.stl')

        cpu_std_json = {
            "closer": closer,
            "further": further,
            "mesh_upper": mesh_upper,
            "mesh_lower": mesh_lower,
            "mesh_jaw": mesh_jaw,
            "mesh_oppo": mesh_oppo,
            "beiya_id": std_out.miss_id,
            "is_single": std_out.is_single,
            "standard": standard,
            "template_name": template_name,
            "cpu_points_info": cpu_points_info,
            "cpu_colors_info": cpu_colors_info,
            "points_oppo_id": points_oppo_id,
        }
        cpu_std_json_backup = {
            "closer": closer,
            "further": further,
            "mesh_upper": mesh_upper,
            "mesh_lower": mesh_lower,
            "mesh_jaw": mesh_jaw,
            "mesh_oppo": mesh_oppo,
            "beiya_id": std_out.miss_id,
            "is_single": std_out.is_single,
            "standard": mesh_backup,
            "template_name": template_name,
            "cpu_points_info": cpu_points_info_backup,
            "cpu_colors_info_backup": cpu_colors_info_backup,
            "points_oppo_id": points_oppo_id,
        }
        std_json = {
            "cpu_std_json": cpu_std_json,
            # "cpu_std_json_backup": cpu_std_json_backup,
            "standard": standard,
            "inner": mesh_beiya,
            "standard_backup": mesh_backup,
        }

        print("suncess stdcrown")

        # 本地rie测试保存结果
        # with open('std0828.json', "w") as f:
        #     f.write(json.dumps(std_json))

        return {"Msg": {"data": std_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {"Msg": traceback.format_exc(), "Code": 203, "State": "Failure"}
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import os
    import time

    # save_dir = r"/home/wanglong/PycharmProjects/lambda_crown/cad_git/download"
    # cases = os.listdir(save_dir)
    # for case_id in cases:
    #     case_id = "0c4f68c4-43e1-4efd-8547-4f17ec30a32e"
    #     print(case_id)
    #     with open(os.path.join(save_dir, case_id, "gpu-step-2-result.json"), "r") as f:
    #         event_ = json.load(f)
    #     event = event_["cpu_process_info"]
    #     s1 = time.time()
    #     out = handler(event, os.path.join(save_dir, case_id))
    #     s2 = time.time()
    #     out["Msg"]["data"]['prep_extended'] = event_['prep_extended']
    #     print(s2 - s1)
    #     with open(os.path.join(save_dir, case_id, "std.json"), "w") as f:
    #         f.write(json.dumps(out["Msg"]["data"]))
    #     break

    save_dir = "test_data"
    case_id = "0617"

    with open(os.path.join(save_dir, case_id, "gpu.json"), "r") as f:
        event = json.load(f)["cpu_process_info"]

    out = handler(event, os.path.join(save_dir, case_id))

    with open(os.path.join(save_dir, case_id, "std1.json"), "w") as f:
        f.write(json.dumps(out["Msg"]["data"]))
    # print
