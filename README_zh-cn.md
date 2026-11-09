# AI 嵌体生成

## English

[Click here for the English version](README.md)

这是一个用于从AI预测生成和后处理算法设计嵌体网格的代码库。


## 已实现

- [x] 可调节的水泥间隙
- [x] 可调节的缝合距离
- [x] 可调节的边缘距离，不会因调整而变形
- [x] 可调节的边缘距离，会因调整而变形
- [x] 可调节的插入深度
- [x] 可调节的边缘距离，不会因最小厚度而膨胀
- [x] 膨胀后具有更好解剖结构的最小厚度
- [x] 支持可调节的最小厚度
- [x] 一个不会偶尔停止工作的稳健的最小厚度算法
- [x] 重构工作流程以受益于更多的牙科先验知识
- [x] 更好的边缘调整，使缝合看起来更好
- [x] 重新设计近端调整算法
- [x] WebUI 部署
- [x] 咬合调整中的稳健碰撞检测
- [x] Studio 交接
- [x] 多嵌体生成
- [x] 术前镜像流程

## 开发中

- [ ] 


## 备注

### 倒凹处理
- **`fill_undercut`**（配置项）: 启用时，在水泥间隙膨胀前对嵌体内冠进行定向倒凹填充。
- **`undercut_filled`**（配置项）: 倒凹填充完成后标记为 `true`。另有一个基于 `pruner_fun` 的修剪方案已实现但当前禁用；该方案在倒凹未填充时用备牙扫描修剪网格。

## 待办事项



## modal部署

### 部署方式:通过deploy_fastapi.py部署，比如:modal deploy --name [app_name] deploy_fastapi.py

### 接口

- mesh_repair
  - 描述：网格修复
  - url： `/mesh_repair`
  - 输入参数：
    - 'mesh_upper': 上颌模型
    - 'mesh_lower'：下颌模型
  - 输出参数：
    - 'mesh_upper': 上颌模型
    - 'mesh_lower'：下颌模型

- stdcrown
  - 描述：术前、镜像流程标准冠生成接口
  - url： `/stdcrown`
  - 输入参数：
    - 公共（两种流程都需要）：
      - 'preop_or_mirror': `"preop"` 走术前形变流程，`"mirror"` 走镜像流程，决定算法路径
      - 'beiya_id': str 备牙ID（FDI），首位 `1`/`2` 选上颌，否则下颌
      - 'std_crown': base64 Draco，GPU 输出的标准冠
    - 术前流程（`preop_or_mirror == "preop"`）额外：
      - 'pre_op_crown': base64 Draco，术前牙冠
      - 'preop_to_postop': dict，含 `upper_matrix`/`lower_matrix`（4*4 变换矩阵，按 `beiya_id` 选取，缺省为单位阵；内部映射为 `preop_matrix`）
    - 镜像流程（`preop_or_mirror == "mirror"`）额外：
      - 'mirror_id': str 镜像牙ID
      - 'all_other_crown': GPU 输出，每颗牙的分割结果，含 `upper_arch`/`lower_arch`
      - 'prep_tid': 备牙 TID
      - 'rot_matrix': list，GPU 输出的分割结果→牙冠局部坐标系的变换矩阵（2 个）
    - 可选：
      - 'test': 默认 `false`，镜像流程为 `true` 时按 `save_path` 落盘 STL
      - 'save_path': 调试落盘目录，默认 `""`
      - 'source_url' / 'target_url': URL 适配器字段，见“URL 适配器支持”
      - 'job_id' / 'execution_id': 仅用于日志
  - 输出参数：
    - 'stdcrown': base64 Draco，生成的标准冠（经术前形变或镜像后，再做各向同性重网格 targetlen 0.15 + 取最大连通域）。作为后续 postprocess 接口的初始牙冠输入。

- postprocess
  - 描述： 根据初始位置生成嵌体
  - url: `/postprocess`
  - 输入参数：
    - 'multi_restoration': 是否是多修复体
    - 'rot_matrix': 坐标变换矩阵
    - 'mesh_beiya': 备牙
    - 'prep_q'： 嵌体内冠（无填充间隙）
    - 'mesh_upper'： 上颌
    - 'mesh_lower'： 下颌
    - 'mesh1'： 近中邻牙
    - 'mesh2'： 远中邻牙
    - 'stdcrown'： 初始牙冠
    - 'beiya_id'： 备牙ID
    - 'inner': （可选）预计算的嵌体内表面（已含粘接间隙 + 厚度处理），base64 Draco 编码，由 `thickness_shell` 路由产出。与 `thickness_shell` 一起传入时，主流程直接复用，跳过填倒凹 + 粘接间隙 + 厚度外壳生成。
    - 'thickness_shell': （可选）预计算的厚度外壳，base64 Draco 编码，由 `thickness_shell` 路由产出。主流程直接复用，不再内部计算。
    - 'fill_undercut': 无效。主流程是否填倒凹取决于是否传入 `inner` + `thickness_shell`（未传则执行），与此开关无关。
    - 'insert_dir': （可选）倒凹插入方向 `[x, y, z]`。未传 `inner` + `thickness_shell` 时用于填倒凹（未传则自动搜索）；传入 `inner` + `thickness_shell` 时忽略。
    - 'paras':
      - 'adjust_crown': 默认为1，若设置为0，直接将stdcrown与膨胀后的prep_q缝合，不做邻接咬合厚度调整
      - 'cement_gap': 覆盖水泥间隙值（configs.yaml 默认值: 0.08）。也可作为顶层字段传入，顶层字段优先级高于 paras。
  - 输出参数：
    - 'crown'： 嵌体
    - 'inlay_outer'： 嵌体外冠
    - 'inner_dilation'： 嵌体内冠（有填充间隙）
    - 'thickness_shell'： 厚度外壳网格

- thickness_shell
  - 描述：只生成厚度外壳（填倒凹 + 粘接间隙 + 厚度外壳）。复用同一个 `postprocess.handler` Lambda 入口，通过 `action: "thickness_shell"` 路由分发。
  - url: `/postprocess`
  - 输入参数：
    - 'action': 必须为 `"thickness_shell"` 以选择此路由（不填 / 其它值 = 完整 postprocess 流程）
    - 'inlay_inner': 嵌体内冠（无填充间隙），base64 Draco 编码，与 postprocess 的 `prep_q` 同源
    - 'insert_dir': （可选）倒凹插入方向 `[x, y, z]`。未传时 `undercut_py` 自动搜索最佳方向
    - 'cement_gap': （可选）粘接间隙值（configs.yaml 默认值: 0.08）。也可放在 `paras.cement_gap` 里，顶层 `cement_gap` 优先级高于 `paras`
  - 输出参数：
    - 'thickness_shell': 厚度外壳网格（base64 Draco 编码）
    - 'inner': 粘接间隙 + 厚度处理后的嵌体内表面（base64 Draco 编码）。回传给 postprocess 路由作为 `inner` 输入。
  - 处理流程：填倒凹（`undercut_py`，始终执行）→ 粘接间隙（`MeshRegistration.get_cement_gap`）→ 厚度外壳（`utils.get_thickness_gap`）。不做 `get_thickness_gap2` 精修，无需 crown/upper/lower 等其它输入。

### URL 适配器支持

`stdcrown.py` 和 `postprocess.py` 的核心 handler 现在支持可选的 URL 适配器流程。
这项能力适用于直接调用 handler，以及 `app.py` 暴露的本地 Flask `/stdcrown`、`/postprocess` 接口。
当前 `deploy_fastapi.py` / `api.py` 这套 Modal 包装接口还没有同步这部分逻辑。

- `/stdcrown` 和 `/postprocess` 可选新增字段：
  - 'source_url'：存在时，handler 会先通过 `GET` 下载完整 JSON 输入。
  - 'target_url'：存在时，handler 会通过 `PUT` 上传最终结果。
- 行为说明：
  - 没有 `source_url` 时，继续直接使用请求体作为输入，兼容原有调用方式。
  - 有 `target_url` 时，业务结果会按 `{"Msg":{"data": ...},"Code":200,"State":"Success"}` 结构上传，同时 handler 自身返回 `{"Msg":{"data":"result saved to target url"},"Code":200,"State":"Success"}`。
  - `target_url` 始终从原始请求 event 读取，不会从下载后的 payload 中取值。
  - 输入兼容处理会在 `source_url` 下载完成后再执行。

- occlusion
  - 描述：对手动调整后的嵌体进行咬合、邻接、厚度调整
  - url: `/occlusion`
  - 输入参数：
    - 'inner_dilation'： 嵌体内冠（有填充间隙）
    - 'mesh_upper'： 上颌
    - 'mesh_lower'： 下颌
    - 'mesh1'： 近中邻牙
    - 'mesh2'： 远中邻牙
    - 'inlay_outer'： 嵌体外冠
    - 'beiya_id'： 备牙ID
  - 输出参数：
    - 'crown'： 嵌体（增厚前）
    - 'inlay_outer'： 嵌体外冠（增厚前）
    - 'fixed_crown': 增厚后的嵌体
    - 'fixed_inlay_outer': 增厚后的外冠

- stitch_edge
  - 描述：缝合外冠和内冠
  - url: `/stitch_edge`
  - 输入参数：
    - 'multi_restoration': 是否是多修复体
    - 'rot_matrix': 坐标变换矩阵
    - 'inner_dilation'： 嵌体内冠（无填充间隙）
    - 'inlay_outer'： 嵌体外冠
  - 输出参数：
    - 'crown'： 嵌体
    - 'inner_dilation'： 嵌体内冠（有填充间隙）

### 返回内容

- 成功：{"Msg":{'data':out_items}, "Code": 200,"State": "Success", "version": "1.0.0"}
- 失败：{"error": error_message}

## 备注
