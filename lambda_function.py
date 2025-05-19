import os
import json
from crown_cpu import t
from crown_cpu import write_mesh_bytes


def handler(event, context):
    out = t(event.get('mesh1'), event.get('mesh2'), event.get('mesh_beiya'), event.get('mesh_upper'),
                    event.get('mesh_lower'), event.get('all_other_crowns'), event.get('kps'),event.get('beiya_id'), event.get(
                        'transform'), event.get('voxel_logits'),
                    event.get('is_single'), event.get('pt1'), event.get('pt2'))
    return write_mesh_bytes(out.vertices, out.faces)
