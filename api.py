import base64

import DracoPy
import MQCompressPy
import numpy as np
import trimesh

from mesh_repair import run as mesh_repair_run
from occlusion import run as occlusion_run
from postprocess import run as postprocess_run
from stdcrown import run as stdcrown_run
from stitch_edge import run as stitch_edge_run


def read_mesh_bytes(buffer):
    a = base64.b64decode(buffer)
    mesh_object = DracoPy.decode_buffer_to_mesh(a)
    V = np.array(mesh_object.points).astype(np.float32).reshape(-1, 3)
    F = np.array(mesh_object.faces).astype(np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F).as_open3d


def write_mesh_bytes(mesh, preserve_order=False, colors=None):
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


def show_job_id(msg):
    if msg.get("job_id"):
        print(f"job_id: {msg.get('job_id')}")
    else:
        print("job_id: None")
    if msg.get("execution_id"):
        print(f"execution_id: {msg.get('execution_id')}")
    else:
        print("execution_id: None")


def stdcrown_msg(msg):
    show_job_id(msg)
    data_input = {}
    data_input["beiya_id"] = msg.get("beiya_id")
    st_out = stdcrown_run(data_input)
    data_output = {
        "Msg": {"data": {"stdcrown": write_mesh_bytes(st_out)}},
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


def postprocess_msg(msg):
    show_job_id(msg)
    data_input = {}
    data_input["prep_tooth"] = read_mesh_bytes(msg.get("mesh_beiya"))
    data_input["inlay_inner"] = read_mesh_bytes(msg.get("prep_q"))
    data_input["upper_scan"] = read_mesh_bytes(msg.get("mesh_upper"))
    data_input["lower_scan"] = read_mesh_bytes(msg.get("mesh_lower"))
    data_input["adjacent_teeth"] = [
        read_mesh_bytes(msg.get("mesh1")),
        read_mesh_bytes(msg.get("mesh2")),
    ]
    data_input["standard"] = read_mesh_bytes(msg.get("stdcrown"))
    data_input["tid"] = int(msg.get("beiya_id"))
    data_input["paras"] = msg.get("paras")
    po_out = postprocess_run(data_input)
    data_output = {
        "Msg": {
            "data": {
                "crown": write_mesh_bytes(po_out[0]),
                "inlay_outer": write_mesh_bytes(po_out[1]),
                "inner_dilation": write_mesh_bytes(po_out[2]),
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


def occlusion_msg(msg):
    show_job_id(msg)
    data_input = {}
    data_input["inner_dilation"] = read_mesh_bytes(msg.get("inner_dilation"))
    data_input["upper_scan"] = read_mesh_bytes(msg.get("mesh_upper"))
    data_input["lower_scan"] = read_mesh_bytes(msg.get("mesh_lower"))
    data_input["adjacent_teeth"] = [
        read_mesh_bytes(msg.get("mesh1")),
        read_mesh_bytes(msg.get("mesh2")),
    ]
    data_input["inlay_outer"] = read_mesh_bytes(msg.get("inlay_outer"))
    data_input["tid"] = int(msg.get("beiya_id"))
    occlu_out = occlusion_run(data_input)
    data_output = {
        "Msg": {
            "data": {
                "crown": write_mesh_bytes(occlu_out["stitched_inlay"]),
                "inlay_outer": write_mesh_bytes(occlu_out["inlay_outer"]),
                "fixed_crown": compress_drc(
                    occlu_out["fixed_stitched_inlay"], occlu_out["thickness_points"]
                ),
                "fixed_inlay_outer": write_mesh_bytes(occlu_out["fixed_inlay_outer"]),
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


def stitch_edge_msg(msg):
    show_job_id(msg)
    data_input = {}
    data_input["inner_dilation"] = read_mesh_bytes(msg.get("inner_dilation"))
    data_input["inlay_outer"] = read_mesh_bytes(msg.get("inlay_outer"))
    stitch_out = stitch_edge_run(data_input)
    data_output = {
        "Msg": {
            "data": {
                "crown": write_mesh_bytes(stitch_out[0]),
                "inner_dilation": write_mesh_bytes(stitch_out[1]),
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


def mesh_repair_msg(msg):
    show_job_id(msg)
    data_input = {}
    data_input["upper"] = read_mesh_bytes(msg.get("mesh_upper"))
    data_input["lower"] = read_mesh_bytes(msg.get("mesh_lower"))
    repaired_out = mesh_repair_run(data_input)
    data_output = {
        "Msg": {
            "data": {
                "mesh_upper": write_mesh_bytes(repaired_out[0]),
                "mesh_lower": write_mesh_bytes(repaired_out[1]),
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


def test_func(msg):
    data_output = {
        "Msg": {
            "data": {
                "info": "success",
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }
    return data_output


if __name__ == "__main__":
    import json
    
    # test_post
    # with open('test_data/642f4b03-dd14-4567-a7be-0a2d4b93cb8b/gpu_9afa9d3e-cb95-48de-be82-14154a2cd7a9/output.json') as f:
    #     data = json.load(f)['cpu_process_info']

    # with open('test_data/642f4b03-dd14-4567-a7be-0a2d4b93cb8b/post_24e59bdd-04f2-4a7f-96c0-5bb6a7d7959b/input.json', ) as f:
    #     data_out = json.load(f)

    # data_input = data
    # for k,v in data_out.items():
    #     data_input[k] = v

    # out = postprocess_msg(data_input)

    # with open('postprocess.json', 'w') as f:
    #     json.dump(out, f)
    # print(out)
    
    # test_occlusion
    # with open('test_data/844_45ed13bf-4722-4622-b36f-1aaf1cc6d23e/gpu_d83ef0c7-0f38-4b81-b02c-88e4955b6d24/output.json') as f:
    #     data = json.load(f)['cpu_process_info']

    # with open('test_data/844_45ed13bf-4722-4622-b36f-1aaf1cc6d23e/occ_7aa7b131-3edc-4b64-8959-95c2f0e68d76/output.json', ) as f:
    #     data_out = json.load(f)

    # with open('test_data/844_45ed13bf-4722-4622-b36f-1aaf1cc6d23e/occ_7aa7b131-3edc-4b64-8959-95c2f0e68d76/input.json', ) as f:
    #     data_input = json.load(f)

    # for k,v in data.items():
    #     if k == 'inlay_outer':
    #         continue
    #     data_input[k] = v
    # for k,v in data_out.items():
    #     if k == 'inlay_outer':
    #         continue
    #     data_input[k] = v

    # out = occlusion_msg(data_input)

    # with open('postprocess.json', 'w') as f:
    #     json.dump(out, f)
    # print(out)
    

    # test_stitch
    # with open(
    #     "test_data/642f4b03-dd14-4567-a7be-0a2d4b93cb8b/post_24e59bdd-04f2-4a7f-96c0-5bb6a7d7959b/output.json"
    # ) as f:
    #     data = json.load(f)

    # with open(
    #     "test_data/642f4b03-dd14-4567-a7be-0a2d4b93cb8b/stitch_15b233fd-304f-4a1d-9ba5-4112fe865183/input.json",
    # ) as f:
    #     data_out = json.load(f)

    # data_input = {}
    # data_input["inner_dilation"] = data_out["inner_dilation"]
    # data_input["inlay_outer"] = data["inlay_outer"]

    # out = stitch_edge_msg(data_input)

    # with open("stitch.json", "w") as f:
    #     json.dump(out, f)
    # print(out)
