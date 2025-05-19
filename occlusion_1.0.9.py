import base64
import DracoPy
import numpy as np
import trimesh
import json
from crown_cpu import occlu
import time

def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)

def write_mesh_bytes(mesh):
    encoding_test = DracoPy.encode_mesh_to_buffer(mesh.vertices, mesh.faces)
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode('utf-8')
    return b64_str

def write_drc(drc, file):
    with open(file, 'wb') as test_file:
        test_file.write(base64.b64decode(drc))

def read_drc(file_name):
    """read .drc or .mq file
    Args:
        file_name ([a directory]): [description]
    Returns:
        verctics and faces in np.array (n,3), (m,3)
    """
    with open(file_name, 'rb') as draco_file:
        file_content = draco_file.read()
    b64_bytes = base64.b64encode(file_content)
    b64_str = b64_bytes.decode('utf-8')
    return b64_str
        
def handler(event, context):
    print("receive case")
    # try:
    print('start AI_Crown_Occlu ..')
    cpu_input_json = {
        "out":event.get('out'),
        "inner":event.get('inner'),
        "paras":event.get('paras')
    }
    for k, v in event['cpu_post_json'].items():
        event[k] = v
    occlu_out = occlu(event)
    occlu_out.mesh.export(os.path.join(context, 'ai_scan_', 'occlu.stl'))
    crown = write_mesh_bytes(occlu_out.mesh)
    out = write_mesh_bytes(occlu_out.mesh_outside)
    inner = write_mesh_bytes(occlu_out.mesh_beiya)
    occlu_json = {
        'crown':crown,
        'out':out,
        'inner':inner,
        'cpu_input_json':cpu_input_json
    }
    print("suncess occlu")
    
    return {'Msg': {"data": occlu_json}, "Code": 200, "State": "Success"}
    # except Exception as ex:
    #     print(ex)
    #     res = {"Msg":ex, "Code": 203,"State": "Failure"}
    #     res = json.dumps(res)
    #     return res
    

if __name__ == "__main__":
    import time
    import os
    save_dir = r'/media/wanglong/Elements/wanglong/3shape_1018'
    case_id = '1ae8-7235'
    with open(os.path.join(save_dir, case_id, 'post.json'), 'r') as f:
        event = json.load(f)
    s1 = time.time()
    out = handler(event, os.path.join(save_dir, case_id))
    s2 = time.time()
    print(s2 - s1)