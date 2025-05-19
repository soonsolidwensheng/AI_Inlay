import traceback

from inlay_cpu import InlayGeneration
from utils import compress_drc, read_mesh_bytes, write_mesh_bytes


def run(data):
    IG = InlayGeneration(
        tid=data.get("tid"),
        prep_tooth=data.get("prep_tooth"),
        inlay_inner=data.get("inner_dilation"),
        upper_scan=data.get("upper_scan"),
        lower_scan=data.get("lower_scan"),
        adjacent_teeth=[x for x in data.get("adjacent_teeth")],
        standard=data.get("inlay_outer"),
    )
    IG.run_occlu()

    return {
        "stitched_inlay": IG.o3d2tri(IG.stitched_inlay),
        "inlay_outer": IG.o3d2tri(IG.inlay_outer),
        "fixed_stitched_inlay": IG.fixed_stitched_inlay,
        "fixed_inlay_outer": IG.fixed_inlay_outer,
        "thickness_points": IG.thickness_points_id,
    }


def handler(event, context):
    print("receive case")
    try:
        print("start AI_Inlay_Occlu ..")
        if event.get("job_id"):
            print(f"job_id: {event.get('job_id')}")
        else:
            print("job_id: None")
        if event.get("execution_id"):
            print(f"execution_id: {event.get('execution_id')}")
        else:
            print("execution_id: None")
        data_input = {}
        data_input["inner_dilation"] = read_mesh_bytes(event.get("inner_dilation"))
        data_input["upper_scan"] = read_mesh_bytes(event.get("mesh_upper"))
        data_input["lower_scan"] = read_mesh_bytes(event.get("mesh_lower"))
        data_input["adjacent_teeth"] = [
            read_mesh_bytes(event.get("mesh1")),
            read_mesh_bytes(event.get("mesh2")),
        ]
        data_input["inlay_outer"] = read_mesh_bytes(event.get("inlay_outer"))
        data_input["tid"] = int(event.get("beiya_id"))
        occlu_out = run(data_input)
        occlu_json = {
            "crown": write_mesh_bytes(occlu_out["stitched_inlay"]),
            "inlay_outer": write_mesh_bytes(occlu_out["inlay_outer"]),
            "fixed_crown": compress_drc(
                occlu_out["fixed_stitched_inlay"], occlu_out["thickness_points"]
            ),
            "fixed_inlay_outer": write_mesh_bytes(occlu_out["fixed_inlay_outer"]),
            "modal_function_call_id": None,
        }
        print("suncess occlusion")

        return {"Msg": {"data": occlu_json}, "Code": 200, "State": "Success"}
    except Exception as _:
        res = {
            "error": traceback.format_exc(),
            "modal_function_call_id": None,
        }
        traceback.print_exc()
        return res
