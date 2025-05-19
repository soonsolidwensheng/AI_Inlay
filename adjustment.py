import base64
import DracoPy
import numpy as np
import trimesh
import json
from crown_cpu import adjust

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

def handler(event, context):
    print("receive case")
    try:
        print('start AI_Crown_Adjust ..')
        adjust_out = adjust(event)
        crown = write_mesh_bytes(adjust_out.mesh)
        out = write_mesh_bytes(adjust_out.mesh_outside)
        inner = write_mesh_bytes(adjust_out.mesh_beiya)
        adjust_json = {
            'crown':crown,
            'out':out,
            'inner':inner,
        }
        print("suncess adjust")
        
        return {'Msg': {"data": adjust_json}, "Code": 200, "State": "Success"}
    except Exception as ex:
        print(ex)
        res = {"Msg":ex, "Code": 203,"State": "Failure"}
        res = json.dumps(res)
        return res
        

        
        
