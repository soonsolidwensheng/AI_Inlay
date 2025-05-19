import json
import base64
import traceback
import DracoPy
import numpy as np
import trimesh
from crown_cpu import stitch_edge


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


def write_mesh_bytes(mesh, preserve_order=False, colors=None):
    # 设置 Draco 编码选项
    encoding_test = DracoPy.encode_mesh_to_buffer(mesh.vertices, mesh.faces, preserve_order=preserve_order, quantization_bits=14,
                                                  compression_level=10, colors=colors)
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def write_drc(drc, file):
    with open(file, "wb") as test_file:
        test_file.write(base64.b64decode(drc))


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
        print('start AI_Crown_Stitch_Edge ..')
        # 本地rie测试输入
        # path = r'/media/wanglong/Elements/wanglong/标准牙体库测试结果/1ae8-9331/Mature v1'
        # inner = trimesh.load(f'{path}/dilation.stl')
        # inner = write_mesh_bytes(inner)
        # outer = trimesh.load(f'{path}/trans_neck.stl')
        # outer = write_mesh_bytes(outer)
        # event["inner"] = inner
        # event["out"] = outer
        # event["align_edges"] = False
        
        cpu_input_json = {
            "inner":event.get('inner'),
            "out":event.get('out'),
            "align_edges":event.get('align_edges'),
        }
        if 'cpu_info_json' in event.keys():
            for k, v in event['cpu_info_json'].items():
                event[k] = v
        else:
            pass
        print("pred_filestem_name", event.get("pred_filestem_name"))
        if event.get("pred_filestem_name"):
            event["template_name"] = file_name2template_name(event.get("pred_filestem_name"))
        print("template_name", event.get('template_name'))
        print("job_id", event.get("job_id"))
        stitch_out = stitch_edge(event)
        crown = write_mesh_bytes(stitch_out.mesh)
        inner = write_mesh_bytes(stitch_out.mesh_beiya)
        # add_points = np.array(stitch_out.add_points).tolist()
        # add_point_normal = stitch_out.add_point_normal.tolist()
        # axis = np.array(stitch_out.axis).tolist()
        
        # stitch_out.mesh.export('stitch.stl')
        
        stitch_json = {
            'crown':crown,
            # "points_info":{
            #     "points": add_points,
            #     "normals": add_point_normal,
            #     "axis": axis,
            # },
            'inner': inner,
            'cpu_input_json':cpu_input_json
        }
        print("suncess stitch_edge")
        
        return {'Msg': {"data": stitch_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {"Msg": traceback.format_exc(), "Code": 203, "State": "Failure"}
        traceback.print_exc()
        return res


if __name__ == "__main__":
    import time
    import os
    # save_dir = r'test_data_'
    # case_id = 'fe2ed'
    # with open(os.path.join(save_dir, case_id, 'crown_post_process (1).json'), 'r') as f:
    #     event = json.load(f)
    # with open(os.path.join(save_dir, case_id, 'std.json'), 'r') as f:
    #     event['mesh_jaw'] = json.load(f)['cpu_std_json']['mesh_jaw']
    # # read_mesh_bytes(event["fixed_crown"]).export('1.stl')
    # # read_mesh_bytes(event["crown"]).export('2.stl')
    # # trimesh.PointCloud(read_mesh_bytes(event["fixed_crown"]).vertices[event['fixed_points']]).export('3.ply')
    # s1 = time.time()
    # out = handler(event, os.path.join(save_dir, case_id))
    # s2 = time.time()
    # print(s2 - s1)
    # with open(os.path.join(save_dir, case_id, 'stitch_edge.json'), "w") as f:
    #     f.write(json.dumps(out["Msg"]["data"]))