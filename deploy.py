from flask import Flask, request
from stdcrown import handler as std_handler
from postprocess import handler as post_handler
from occlusion import handler as occl_handler
from stitch_edge import handler as stitch_handler
from undercut_filling import handler as undercut_handler
from geometric_utils import handler as geometric_handler


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/std_process", methods=["POST"])
def std_process_handler():
    event = request.json
    return std_handler(event, "")


@app.route("/geo_process", methods=["POST"])
def geo_process_handler():
    event = request.json
    return geometric_handler(event, "")


@app.route("/post_process", methods=["POST"])
def post_process_handler():
    event = request.json
    return post_handler(event, "")


@app.route("/occlusion_process", methods=["POST"])
def occlusion_process_handler():
    event = request.json
    return occl_handler(event, "")


@app.route("/stitch_edge_process", methods=["POST"])
def stitch_edge_process_handler():
    event = request.json
    return stitch_handler(event, "")


@app.route("/undercut_process", methods=["POST"])
def undercut_process_handler():
    event = request.json
    return undercut_handler(event, "")


# WSGI入口函数
def handler(environ, start_response):
    return app(environ, start_response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, threaded=False)
