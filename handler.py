import os
import json
import traceback
import open3d
import numpy as np
from register import MeshRegistration
from inlay_cpu import InlayGeneration
from datetime import datetime
print("import open3d, numpy success")


def simple_log(log_fpath, dict):
    if not os.path.exists(log_fpath):
        with open(log_fpath, 'w+') as f:
            json.dump(dict, f, indent=4)
    else:
        with open(log_fpath, 'r') as f:
            data = json.load(f)
        # Add new key-value pairs
        data.update(dict)
        # Write back to file with pretty formatting
        with open(log_fpath, 'w') as f:
            json.dump(data, f, indent=4)


def read_mesh(path: str) -> open3d.geometry.TriangleMesh:
    mesh = open3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def handler(input_dir: str,
            tid: int, 
            prep_tooth: open3d.geometry.TriangleMesh, 
            inlay_inner: open3d.geometry.TriangleMesh,
            upper: open3d.geometry.TriangleMesh,
            lower: open3d.geometry.TriangleMesh) -> open3d.geometry.TriangleMesh:
    '''
    Params:
        S: the standard crown mesh
        B: the prep crown mesh
        Q: the prep surface to be restored by the inlay design

    Returns:
        U + D: the inner and outer surface of the inlay design
    '''
    IG = InlayGeneration(tid, prep_tooth, inlay_inner, upper, lower, save_path=input_dir)
    IG.run()
    inlay_outer = IG.get_inlay_outer()
    stitched_inlay = IG.get_stitched_inlay()
    return inlay_outer, stitched_inlay


def handler_webui(input_dir: str, tid: int) -> None:
    '''
    Params:
        input_dir: the path that contains:
                the prep seg mesh, the inlay inner mesh, the upper and lower meshes
            eg:      15B.stl,           15Q.stl,         upper.stl,    lower.stl

    Returns:
        None, the result "stitched_inlay.stl" will be saved in the input_dir
    '''
    # load params
    prep_tooth = read_mesh(os.path.join('input', input_dir, f'{tid}B.stl'))
    inlay_inner = read_mesh(os.path.join('input', input_dir, f'{tid}Q.stl'))
    upper = read_mesh(os.path.join('input', input_dir, 'Maxillary.stl'))
    lower = read_mesh(os.path.join('input', input_dir, 'Mandibular.stl'))
    adjacent_teeth = []
    if os.path.exists(os.path.join('input', input_dir, f'{tid+1}.stl')):
        adjacent_teeth.append(read_mesh(os.path.join('input', input_dir, f'{tid+1}.stl')))
    if os.path.exists(os.path.join('input', input_dir, f'{tid-1}.stl')):
        adjacent_teeth.append(read_mesh(os.path.join('input', input_dir, f'{tid-1}.stl')))
    
    # run inlay gen
    IG = InlayGeneration(tid, prep_tooth, inlay_inner, upper, lower, adjacent_teeth)
    IG.run()
    # get and save stitched inlay
    inlay_outer = IG.get_inlay_outer()
    stitched_inlay = IG.get_stitched_inlay()
    if not os.path.exists(os.path.join('output', input_dir)):
        os.makedirs(os.path.join('output', input_dir))
    open3d.io.write_triangle_mesh(os.path.join('output', input_dir, 'stitched_inlay.stl'), stitched_inlay)


def handler_webui_2(data_dir ='/output') -> None:
    '''
    Params:
        input_dir: the path that contains:
                the prep seg mesh, the inlay inner mesh, the upper and lower meshes
            eg:      15B.stl,           15Q.stl,         upper.stl,    lower.stl

    Returns:
        None, the result "stitched_inlay.stl" will be saved in the input_dir
    '''
    
    success = False
    try:
        # load params
        now = datetime.now()
        formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
        log_fpath = os.path.join(data_dir, 'cpu_log.json')
        standard = read_mesh(os.path.join(data_dir, 'standard.stl'))
        prep_tooth = read_mesh(os.path.join(data_dir, f'prep_b.stl'))
        inlay_inner = read_mesh(os.path.join(data_dir, f'prep_q.stl'))
        upper = read_mesh(os.path.join(data_dir, 'upper.stl'))
        lower = read_mesh(os.path.join(data_dir, 'lower.stl'))
        with open(os.path.join(data_dir, 'gpu_result.json'), 'r') as f:
            gpu_res = json.load(f)
        tid = int(gpu_res['cpu_process_info']['beiya_id'])
        adjacent_teeth = []
        if os.path.exists(os.path.join(data_dir, f'{tid+1}.stl')):
            adjacent_teeth.append(read_mesh(os.path.join(data_dir, f'{tid+1}.stl')))
        if os.path.exists(os.path.join(data_dir, f'{tid-1}.stl')):
            adjacent_teeth.append(read_mesh(os.path.join(data_dir, f'{tid-1}.stl')))
        
        # run inlay gen
        IG = InlayGeneration(tid, prep_tooth, inlay_inner, upper, lower, adjacent_teeth, standard, save_path=os.path.join(data_dir, 'intermediate_results'))
        IG.run()
        # get and save stitched inlay
        inlay_outer = IG.get_inlay_outer()
        stitched_inlay = IG.get_stitched_inlay()
        open3d.io.write_triangle_mesh(os.path.join(data_dir, 'sub_mesh.stl'), stitched_inlay)
        success = True
        simple_log(log_fpath, {formatted_date:{'success': success, 'exception': None}})
    except:
        simple_log(log_fpath, {formatted_date:{'success': success, 'exception': traceback.format_exc()}})
        traceback.print_exc()

if __name__ == '__main__':
    handler_webui_2()
    # handler_webui_2('/media/chuanbo/DATA2/data/嵌体数据/designer_Zheng_20230202_1158_增奇口腔_袁_10922266')
    # handler_webui('55ea2904-10c7-4f7b-a95f-d15bb7dc6865', 15)