#####################################################
# Author: Pele Wang
# Date: 2025-01-19
# Description: This scipt contains geometric utility 
#              functions to be called by AWS backend.
#####################################################
import sys
import json
import base64
import trimesh
import DracoPy
import traceback
import numpy as np


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F)


def write_mesh_bytes(mesh, colors=None, preserve_order=False):
    encoding_test = DracoPy.encode_mesh_to_buffer(
        mesh.vertices,
        mesh.faces,
        preserve_order=preserve_order,
        quantization_bits=14, # DracoPy encode settings
        compression_level=10,
        colors=colors,
    )
    b64_bytes = base64.b64encode(encoding_test)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def transform(event):
    '''
    parameters: {
                "util_name": "transform",
                "mesh": base64 mesh,
                "trans_mat": 4x4 transformation matrix,
                }
    returns: base64 transformed mesh
    '''
    mesh = read_mesh_bytes(event["mesh"])
    trans_mat = np.array(event["trans_mat"])
    mesh.apply_transform(trans_mat)
    return mesh


def handler(event, context=None):
    '''
    supported method names:
        1. transform
        2. TBD...
    '''
    print("receive case")
    try:
        method_name = event["util_name"]
        method = getattr(sys.modules[__name__], method_name)
        result = method(event)
        geo_utils_json = {
            "input": event,
            "output": write_mesh_bytes(result),
            "State": "Success",
        }
        print("success stdcrown")
        return geo_utils_json
    except:
        traceback.print_exc()
        return {"Msg": traceback.format_exc(), "State": "Failure"}


if __name__ == "__main__":
    import os

    with open("crown_cpu/test_data/0617/geometric_utils_input.json", "r") as f:
        in_msg = json.load(f)

    out_msg = handler(in_msg)

    with open("crown_cpu/test_data/0617/geometric_utils_output.json", "w") as f:
        json.dump(out_msg, f, indent=4)

    transformed_mesh = read_mesh_bytes(out_msg["output"])
    transformed_mesh.show()
