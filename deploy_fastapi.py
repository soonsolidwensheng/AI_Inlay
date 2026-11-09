import traceback

import modal
from fastapi import FastAPI, Request

from api import (
    mesh_repair_msg,
    occlusion_msg,
    postprocess_msg,
    stdcrown_msg,
    stitch_edge_msg,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update")
    .run_commands("apt-get -y install xterm xauth openssh-server tmux wget")
    .run_commands("apt-get install ffmpeg libsm6 libxext6 -y")
    .run_commands("apt-get update && apt-get install libgl1")
    .run_commands("apt-get clean")
    .run_commands("rm -rf /var/lib/apt/lists/*")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi[standard]")
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=[
            "__pycache__",
            ".git/",
            "test_data/",
            "result/",
            ".vscode/",
            "object_print3d_utils/",
            ".gitignore",
            "test_*.py",
        ],
    )
    .add_local_dir(
        "object_print3d_utils",
        remote_path="/usr/lib/python3.11/site-packages/bpy/4.2/scripts/addons/object_print3d_utils",
    )
)
web_app_test = FastAPI()
web_app_mesh_repair = FastAPI()
web_app_stdcrown = FastAPI()
web_app_postprocess = FastAPI()
web_app_occlusion = FastAPI()
web_app_stitch_edge = FastAPI()
app = modal.App("Inlay_Onlay_CPU", image=image)


@web_app_mesh_repair.post("/mesh_repair")
async def mesh_repair(request: Request):
    body = await request.json()
    try:
        result = mesh_repair_msg(body)
        result["Msg"]["data"]["modal_function_call_id"] = (
            modal.current_function_call_id()
        )
    except Exception:
        print(traceback.format_exc())
        result = {
            "error": traceback.format_exc(),
            "modal_function_call_id": modal.current_function_call_id(),
        }
    return result


@web_app_stdcrown.post("/stdcrown")
async def stdcrown(request: Request):
    body = await request.json()
    try:
        result = stdcrown_msg(body)
        result["Msg"]["data"]["modal_function_call_id"] = (
            modal.current_function_call_id()
        )
    except Exception:
        print(traceback.format_exc())
        result = {
            "error": traceback.format_exc(),
            "modal_function_call_id": modal.current_function_call_id(),
        }
    return result


@web_app_postprocess.post("/postprocess")
async def postprocess(request: Request):
    body = await request.json()
    try:
        result = postprocess_msg(body)
        result["Msg"]["data"]["modal_function_call_id"] = (
            modal.current_function_call_id()
        )
    except Exception:
        print(traceback.format_exc())
        result = {
            "error": traceback.format_exc(),
            "modal_function_call_id": modal.current_function_call_id(),
        }
    return result


@web_app_occlusion.post("/occlusion")
async def occlusion(request: Request):
    body = await request.json()
    try:
        result = occlusion_msg(body)
        result["Msg"]["data"]["modal_function_call_id"] = (
            modal.current_function_call_id()
        )
    except Exception:
        print(traceback.format_exc())
        result = {
            "error": traceback.format_exc(),
            "modal_function_call_id": modal.current_function_call_id(),
        }
    return result


@web_app_stitch_edge.post("/stitch_edge")
async def stitch_edge(request: Request):
    body = await request.json()
    try:
        result = stitch_edge_msg(body)
        result["Msg"]["data"]["modal_function_call_id"] = (
            modal.current_function_call_id()
        )
    except Exception:
        print(traceback.format_exc())
        result = {
            "error": traceback.format_exc(),
            "modal_function_call_id": modal.current_function_call_id(),
        }
    return result


@web_app_test.post("/test_303")
async def test_303(request: Request):
    # import time
    body = await request.json()
    print("app start")
    # time.sleep(200)
    return {
        "Msg": {
            "data": {
                "info": "success",
                "function_call_id": modal.current_function_call_id(),
            }
        },
        "Code": 200,
        "State": "Success",
        "version": "1.0.0",
    }


@app.function(cpu=4.0, memory=1024, scaledown_window=120, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def test_303_app():
    return web_app_test


@app.function(cpu=16.0, memory=10240, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def mesh_repair_app():
    return web_app_mesh_repair


@app.function(cpu=16.0, memory=10240, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def stdcrown_app():
    return web_app_stdcrown


@app.function(cpu=16.0, memory=10240, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def postprocess_app():
    return web_app_postprocess


@app.function(cpu=16.0, memory=10240, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def occlusion_app():
    return web_app_occlusion


@app.function(cpu=16.0, memory=10240, enable_memory_snapshot=True)
@modal.asgi_app(requires_proxy_auth=True)
def stitch_edge_app():
    return web_app_stitch_edge
