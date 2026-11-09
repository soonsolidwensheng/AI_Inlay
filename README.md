# AI Inlay Generation

## 简体中文

[点击这里查看简体中文版本](README_zh-cn.md)

This is the code base for generation and post processing algorithms that design a Inlay mesh from AI predictions.  

## Implemented

- [x] Adjustable cement gap spacing  
- [x] Adjustable stitching distance
- [x] Adjustable distance to the margin that do not get deformed by adaptations  
- [x] Adjustable distance to the margin that get deformed by adaptations  
- [x] Adjustable insertion depth of adaptations  
- [x] Adjustable distance to the margin that do not get inflated by min thickness
- [x] Min thickness with better anotomy after inflation
- [x] Support for Adjustable min thickness
- [x] A robust min thickness algo that does not stop working occasionally
- [x] Refactoring of the workflow to benefit from more dental prior knowledge  
- [x] Better margin adaptation so that the stitching looks better
- [x] Redesign of the approximal adaptation algorithm
- [x] WebUI deployment
- [x] Robust collision detection in occlusal adaptation
- [x] Studio handoff

## Under Development

- [ ] Generation of two adjacent inlays (in terms of adjacency relationships)

## Notes

### Undercut Handling
- **`fill_undercut`** (config): When enabled, performs directional undercut filling on the inlay inner mesh before cement gap expansion.
- **`undercut_filled`** (config): When `fill_undercut` completes, this flag is set to `true`. An alternative pruner-based approach (`pruner_fun`) is implemented but currently disabled; it would trim meshes against the prep scan when undercut is not filled.

## TODOs


## Modal Deployment

### Deployment Method: Deploy via deploy_fastapi.py, e.g., modal deploy --name [app_name] deploy_fastapi.py

### Endpoints

- mesh_repair
  - Description: Mesh repair
  - URL: `https://modal--example-name-mesh-repair-app.model.run/mesh_repair`
  - Input Parameters:
    - 'mesh_upper': Upper jaw model
    - 'mesh_lower': Lower jaw model
  - Output Parameters:
    - 'mesh_upper': Upper jaw model
    - 'mesh_lower': Lower jaw model

- stdcrown
  - Description: Standard crown generation endpoint (pre-op morphing or mirror flow)
  - URL: `https://modal--example-name-stdcrown-app.model.run/stdcrown`
  - Input Parameters:
    - Common (both flows):
      - 'preop_or_mirror': `"preop"` for pre-op morphing flow, `"mirror"` for mirror flow. Selects the algorithm path.
      - 'beiya_id': Preparation tooth ID (FDI). First char `1`/`2` selects the upper arch, otherwise the lower arch.
      - 'std_crown': Standard crown from GPU output, base64 Draco.
    - Pre-op flow (when `preop_or_mirror == "preop"`):
      - 'pre_op_crown': Pre-op crown, base64 Draco.
      - 'preop_to_postop': Dict with `upper_matrix` and/or `lower_matrix` (4x4 transform). Selected by `beiya_id`; defaults to identity if omitted. (Internally mapped to `preop_matrix`.)
    - Mirror flow (when `preop_or_mirror == "mirror"`):
      - 'mirror_id': Mirror target tooth ID.
      - 'all_other_crown': Per-tooth segmentation from GPU, with `upper_arch`/`lower_arch` keys.
      - 'prep_tid': Preparation tooth TID.
      - 'rot_matrix': List of 2 matrices from GPU (segmentation→tooth-local transform).
    - Optional:
      - 'test': Default `false`. Mirror flow saves an STL to `save_path` when `true`.
      - 'save_path': Debug save directory, default `""`.
      - 'source_url' / 'target_url': URL adapter fields, see "URL Adapter Support".
      - 'job_id' / 'execution_id': Logging only.
  - Output Parameters:
    - 'stdcrown': Generated standard crown, base64 Draco. Produced by pre-op morphing or mirror, then isotropic remeshing (targetlen 0.15) and largest-mesh selection. This is the standard crown fed into the postprocess endpoint.

- postprocess
  - Description: Generate inlay based on initial position
  - URL: `https://modal--example-name-postprocess-app.model.run/postprocess`
  - Input Parameters:
    - 'multi_restoration': true is multi restoration
    - 'rot_matrix': matrix from gpu
    - 'mesh_beiya': Preparation
    - 'prep_q': Inlay inner crown (without gap filling)
    - 'mesh_upper': Upper jaw
    - 'mesh_lower': Lower jaw
    - 'mesh1': Mesial adjacent tooth
    - 'mesh2': Distal adjacent tooth
    - 'stdcrown': Initial crown
    - 'beiya_id': Preparation ID
    - 'inner': (optional) Pre-computed inlay inner surface (with cement gap + thickness processing), base64 Draco. Produced by the `thickness_shell` route. When supplied with `thickness_shell`, the main flow reuses them directly and skips undercut blockout + cement gap + thickness shell generation.
    - 'thickness_shell': (optional) Pre-computed thickness shell, base64 Draco. Produced by the `thickness_shell` route. Reused directly by the main flow instead of computing internally.
    - 'fill_undercut': No effect. Undercut blockout in the main flow is controlled by whether `inner` + `thickness_shell` are supplied (run when absent), not by this flag.
    - 'insert_dir': (optional) Undercut insertion direction `[x, y, z]`. Used for undercut blockout when `inner` + `thickness_shell` are not supplied (auto-searched if omitted). Ignored when `inner` + `thickness_shell` are supplied.
    - 'paras':
      - 'adjust_crown': Default is 1, if set to 0, directly stitch stdcrown with expanded prep_q without adjacency and occlusal thickness adjustment
      - 'cement_gap': Override for cement gap (configs.yaml default: 0.08). Can also be passed as a top-level field; top-level takes precedence over paras.
  - Output Parameters:
    - 'crown': Inlay
    - 'inlay_outer': Inlay outer crown
    - 'inner_dilation': Inlay inner crown (with gap filling)
    - 'thickness_shell': Thickness shell mesh

- thickness_shell
  - Description: Generate thickness shell only (undercut blockout + cement gap + thickness shell). Reuses the same `postprocess.handler` Lambda entrypoint; dispatch by `action: "thickness_shell"`.
  - URL: `https://modal--example-name-postprocess-app.model.run/postprocess`
  - Input Parameters:
    - 'action': Must be `"thickness_shell"` to select this route (omit / any other value = full postprocess)
    - 'inlay_inner': Inlay inner crown (without gap filling), base64 Draco. Same source as postprocess `prep_q`.
    - 'insert_dir': (optional) Undercut insertion direction `[x, y, z]`. If omitted, `undercut_py` auto-searches the best direction.
    - 'cement_gap': (optional) Cement gap value (configs.yaml default: 0.08). Can also be passed inside `paras.cement_gap`; top-level `cement_gap` takes precedence over `paras`.
  - Output Parameters:
    - 'thickness_shell': Thickness shell mesh (base64 Draco)
    - 'inner': Inlay inner surface after cement-gap + thickness processing (base64 Draco). Feed this back into the postprocess route as the `inner` input.
  - Pipeline: undercut blockout (`undercut_py`, always) → cement gap (`MeshRegistration.get_cement_gap`) → thickness shell (`utils.get_thickness_gap`). No `get_thickness_gap2` refine; no crown/upper/lower needed.

### URL Adapter Support

The core handlers in `stdcrown.py` and `postprocess.py` now support an optional URL adapter flow.
This applies to direct handler calls and the local Flask routes in `app.py`.
It does not apply to the current `deploy_fastapi.py` / `api.py` Modal wrapper endpoints yet.

- Optional input fields for `/stdcrown` and `/postprocess`:
  - 'source_url': If present, the handler downloads the full JSON input from this URL with `GET`.
  - 'target_url': If present, the handler uploads the final result to this URL with `PUT`.
- Behavior:
  - When `source_url` is absent, the handler uses the request body directly, which is backward compatible with the previous flow.
  - When `target_url` is present, the business result is uploaded as `{"Msg":{"data": ...},"Code":200,"State":"Success"}` and the handler returns `{"Msg":{"data":"result saved to target url"},"Code":200,"State":"Success"}`.
  - `target_url` is always read from the original request event, not from the downloaded payload.
  - Input normalization happens after `source_url` is resolved.

- occlusion
  - Description: Perform occlusion, adjacency, and thickness adjustments on manually adjusted inlay
  - URL: `https://modal--example-name-occlusion-app.model.run/occlusion`
  - Input Parameters:
    - 'inner_dilation': Inlay inner crown (with gap filling)
    - 'mesh_upper': Upper jaw
    - 'mesh_lower': Lower jaw
    - 'mesh1': Mesial adjacent tooth
    - 'mesh2': Distal adjacent tooth
    - 'inlay_outer': Inlay outer crown
    - 'beiya_id': Preparation ID
  - Output Parameters:
    - 'crown': Inlay (before thickening)
    - 'inlay_outer': Inlay outer crown (before thickening)
    - 'fixed_crown': Thickened inlay
    - 'fixed_inlay_outer': Thickened outer crown

- stitch_edge
  - Description: Stitch outer crown and inner crown
  - URL: `https://modal--example-name-stitch-edge-app.model.run/stitch_edge`
  - Input Parameters:
    - 'multi_restoration': true is multi restoration
    - 'rot_matrix': matrix from gpu
    - 'inner_dilation': Inlay inner crown (without gap filling)
    - 'inlay_outer': Inlay outer crown
  - Output Parameters:
    - 'crown': Inlay
    - 'inner_dilation': Inlay inner crown (with gap filling)

### Response Content

- Success: {"Msg":{'data':out_items}, "Code": 200,"State": "Success", "version": "1.0.0"}

- Failure: {"error": error_message}

## Notes
