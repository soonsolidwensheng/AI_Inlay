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

## Under Development

- [ ] Studio handoff

## TODOs

- [ ] Generation of two adjacent inlays (in terms of adjacency relationships)

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
  - Description: Standard crown generation endpoint
  - URL: `https://modal--example-name-stdcrown-app.model.run/stdcrown`
  - Input Parameters:
    - 'beiya_id': Preparation ID
  - Output Parameters:
    - 'stdcrown': Initial crown position

- postprocess
  - Description: Generate inlay based on initial position
  - URL: `https://modal--example-name-postprocess-app.model.run/postprocess`
  - Input Parameters:
    - 'mesh_beiya': Preparation
    - 'prep_q': Inlay inner crown (without gap filling)
    - 'mesh_upper': Upper jaw
    - 'mesh_lower': Lower jaw
    - 'mesh1': Mesial adjacent tooth
    - 'mesh2': Distal adjacent tooth
    - 'stdcrown': Initial crown
    - 'beiya_id': Preparation ID
    - 'paras':
      - 'adjust_crown': Default is 1, if set to 0, directly stitch stdcrown with expanded prep_q without adjacency and occlusal thickness adjustment
  - Output Parameters:
    - 'crown': Inlay
    - 'inlay_outer': Inlay outer crown
    - 'inner_dilation': Inlay inner crown (with gap filling)

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
    - 'inner_dilation': Inlay inner crown (without gap filling)
    - 'inlay_outer': Inlay outer crown
  - Output Parameters:
    - 'crown': Inlay
    - 'inner_dilation': Inlay inner crown (with gap filling)

### Response Content

- Success: {"Msg":{'data':out_items}, "Code": 200,"State": "Success", "version": "1.0.0"}

- Failure: {"error": error_message}

## Notes
