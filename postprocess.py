import base64
import copy
import json
import traceback
import DracoPy
import numpy as np
import trimesh
import trimesh.scene

from crown_cpu import post


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


def write_mesh_bytes(mesh, preserve_order=False):
    # 设置 Draco 编码选项
    encoding_test = DracoPy.encode_mesh_to_buffer(
        mesh.vertices,
        mesh.faces,
        preserve_order=preserve_order,
        quantization_bits=14,
        compression_level=10,
        colors=None,
    )
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def write_drc(drc, file):
    with open(file, "wb") as test_file:
        test_file.write(base64.b64decode(drc))


def dict_np2list(data):
    d = copy.deepcopy(data)

    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, dict):
            d[k] = dict_np2list(v)

    return d

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
        print("start AI_Crown_Post ..")

        # 本地rie测试输入
        # event["paras"] = {
        #     "trans_matrix": [1,0,0,0,0,1,0,2.0896362116122376,0,0,1,0,0,0,0,1],
        # }
        # print("event keys", event.keys())
        # for k, v in event.get("params", {}).items():
        #     print(k)
        # event["paras"] = event.get("params", {})
        print("pred_filestem_name", event.get("pred_filestem_name"))
        if event.get("pred_filestem_name"):
            event["template_name"] = file_name2template_name(event.get("pred_filestem_name"))
        cpu_input_json = {
            "inner": event.get("inner"),
            "standard": event.get("standard"),
            "paras": event.get("paras"),
            "crown_rot_matirx": event.get("crown_rot_matirx"),
            "template_name": event.get("template_name", "st_tooth"),
            "points_info": event.get("points_info"),
        }
        if "cpu_undercut_json" in event.keys():
            for k, v in event["cpu_undercut_json"].items():
                if k not in event.keys():
                    event[k] = v
            event["pre_tag"] = "undercut"
        elif "cpu_std_json" in event.keys():
            for k, v in event["cpu_std_json"].items():
                if k not in event.keys():
                    event[k] = v
            event["pre_tag"] = "std"
        print("template_name", event.get("template_name"))
        print("job_id", event.get("job_id"))
        # undercut_mesh = trimesh.load(os.path.join(context, 'vox.stl'))
        # event['undercut_mesh'] = undercut_mesh
        post_out = post(event)
        # if post_out.linya_points is None:
        #     linya_points = np.array(post_out.add_points).tolist()
        # else:
        #     linya_points = np.concatenate([np.array(post_out.linya_points), np.array(post_out.add_points)], axis=0).tolist()
        # post_out.mesh.export(os.path.join(context, 'out.stl'))
        # post_out.undercut_mesh.export(os.path.join(context, 'undercut_out.stl'))
        # post_out.mesh_outside.export('out.stl')
        crown = write_mesh_bytes(post_out.mesh)
        out = write_mesh_bytes(post_out.mesh_outside)
        inner = write_mesh_bytes(post_out.mesh_beiya)
        closer = write_mesh_bytes(post_out.mesh1)
        further = write_mesh_bytes(post_out.mesh2)
        # read_mesh_bytes(crown).export('3.stl')
        cpu_points_info = {}
        points_keys = [
            x for x in vars(post_out.mesh) if x not in vars(trimesh.Trimesh())
        ][2:]
        for key in points_keys:
            points = getattr(post_out.mesh, key, None)
            if points:
                cpu_points_info[key] = points.pt.tolist()
        # adj_points1 = np.array(post_out.adj_points1).tolist()
        # adj_points2 = np.array(post_out.adj_points2).tolist()
        # cross_points = np.array(post_out.mesh.vertices[post_out.cross_points]).tolist()
        # cross_points = np.array(post_out.cross_points).tolist()
        # add_points = np.array(post_out.add_points).tolist()
        # add_point_normal = post_out.add_point_normal.tolist()

        print_points = (
            trimesh.PointCloud(post_out.mesh.ad_points.pt)
            .apply_transform(post_out.print_matrix)
            .vertices.tolist()
        )
        print_normal = (
            trimesh.PointCloud(post_out.add_point_normal)
            .apply_transform(post_out.print_matrix)
            .vertices.tolist()
        )
        axis = np.array(post_out.axis).tolist()
        print_matrix = np.array(post_out.print_matrix).tolist()
        template_name = post_out.template_name

        # post_out.mesh.export(os.path.join(context,'mesh.stl'))
        # post_out.mesh_beiya.export('beiya.stl')
        # post_out.mesh_jaw.export('jaw.stl')
        # trimesh.PointCloud(adj_points1).export('adj_points1.ply')
        # trimesh.PointCloud(adj_points2).export('adj_points2.ply')
        # trimesh.PointCloud(cross_points).export('cross_points.ply')
        # trimesh.PointCloud(add_points).export('add_points.ply')

        cpu_info_json = {
            "mesh_oppo": write_mesh_bytes(post_out.mesh_oppo),
            "closer": closer,
            "further": further,
            "beiya_id": post_out.miss_id,
            "is_single": post_out.is_single,
            "cpu_points_info": cpu_points_info,
            "axis": axis,
            "template_name": template_name,
        }

        post_json = {
            "crown": crown,
            "out": out,
            "inner": inner,
            "points_info": {
                "points": print_points,
                "normals": print_normal,
                "axis": axis,
                "matrix": print_matrix,
            },
            "cpu_info_json": cpu_info_json,
            "cpu_input_json": cpu_input_json,
        }

        print("suncess postprocess")

        # 本地rie测试保存结果
        # with open('post0828.json', "w") as f:
        #     f.write(json.dumps(post_json))

        return {"Msg": {"data": post_json}, "Code": 200, "State": "Success"}
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
    #     # if os.path.exists(os.path.join(save_dir, case_id, 'out.stl')):
    #     #     continue
    #     case_id = "6550a184-a765-4a94-9500-ad3936327208"
    #     print(case_id)
    #     with open(os.path.join(save_dir, case_id, "std.json"), "r") as f:
    #         event = json.load(f)
    #     s1 = time.time()
    #     out = handler(event, os.path.join(save_dir, case_id))
    #     s2 = time.time()
    #     print(s2 - s1)
    #     with open(os.path.join(save_dir, case_id, "post.json"), "w") as f:
    #         f.write(json.dumps(out["Msg"]["data"]))
    #     break
    
    with open('test_data/0617/output.json', "r") as f:
        event = json.load(f)
    for k, v in event['cpu_input_json'].items():
        event[k] = v
    
    out = handler(event, 'test_data')
    with open(os.path.join('test_data', 'post.json'), "w") as f:
        f.write(json.dumps(out["Msg"]["data"]))
    print