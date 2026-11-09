import traceback

import modal

from api import occlusion_msg, postprocess_msg, stdcrown_msg, stitch_edge_msg

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update")
    .run_commands("apt-get -y install xterm xauth openssh-server tmux wget")
    .run_commands("apt-get install ffmpeg libsm6 libxext6 -y")
    .run_commands("apt-get update && apt-get install libgl1")
    .run_commands("apt-get clean")
    .run_commands("rm -rf /var/lib/apt/lists/*")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
    .add_local_dir(
        "object_print3d_utils",
        remote_path="/usr/lib/python3.11/site-packages/bpy/4.2/scripts/addons/object_print3d_utils",
    )
)

app = modal.App("Inlay_Onlay_CPU", image=image)


@app.function(cpu=16.0, memory=10240)
@modal.web_endpoint(method="POST")
def stdcrown(data: dict) -> dict:
    try:
        result = stdcrown_msg(data)
    except Exception:
        result = {"error": traceback.format_exc()}
    return result


@app.function(cpu=16.0, memory=10240)
@modal.web_endpoint(method="POST")
def postprocess(data: dict) -> dict:
    try:
        result = postprocess_msg(data)
    except Exception:
        result = {"error": traceback.format_exc()}
    return result


@app.function(cpu=16.0, memory=10240)
@modal.web_endpoint(method="POST")
def occlusion(data: dict) -> dict:
    try:
        result = occlusion_msg(data)
    except Exception:
        result = {"error": traceback.format_exc()}
    return result


@app.function(cpu=16.0, memory=10240)
@modal.web_endpoint(method="POST")
def stitch_edge(data: dict) -> dict:
    try:
        result = stitch_edge_msg(data)
    except Exception:
        result = {"error": traceback.format_exc()}
    return result
