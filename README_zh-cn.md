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

## 开发中

- [ ] Studio 交接

## 待办事项

- [ ] 连续两颗嵌体的生成（邻接关系方面）

## modal部署

### 部署方式:通过deploy_fastapi.py部署，比如:modal deploy --name [app_name] deploy_fastapi.py

### 接口

- mesh_repair
  - 描述：网格修复
  - url： `https://modal--example-name-mesh-repair-app.model.run/mesh_repair`
  - 输入参数：
    - 'mesh_upper': 上颌模型
    - 'mesh_lower'：下颌模型
  - 输出参数：
    - 'mesh_upper': 上颌模型
    - 'mesh_lower'：下颌模型

- stdcrown
  - 描述：标准冠生成接口
  - url： `https://modal--example-name-stdcrown-app.model.run/stdcrown`
  - 输入参数：
    - 'beiya_id': 备牙ID
  - 输出参数：
    - 'stdcrown': 牙冠初始位置

- postprocess
  - 描述： 根据初始位置生成嵌体
  - url: `https://modal--example-name-postprocess-app.model.run/postprocess`
  - 输入参数：
    - 'mesh_beiya': 备牙
    - 'prep_q'： 嵌体内冠（无填充间隙）
    - 'mesh_upper'： 上颌
    - 'mesh_lower'： 下颌
    - 'mesh1'： 近中邻牙
    - 'mesh2'： 远中邻牙
    - 'stdcrown'： 初始牙冠
    - 'beiya_id'： 备牙ID
    - 'paras':
      - 'adjust_crown': 默认为1，若设置为0，直接将stdcrown与膨胀后的prep_q缝合，不做邻接咬合厚度调整
  - 输出参数：
    - 'crown'： 嵌体
    - 'inlay_outer'： 嵌体外冠
    - 'inner_dilation'： 嵌体内冠（有填充间隙）

- occlusion
  - 描述：对手动调整后的嵌体进行咬合、邻接、厚度调整
  - url: `https://modal--example-name-occlusion-app.model.run/occlusion`
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
  - url: `https://modal--example-name-stitch-edge-app.model.run/stitch_edge`
  - 输入参数：
    - 'inner_dilation'： 嵌体内冠（无填充间隙）
    - 'inlay_outer'： 嵌体外冠
  - 输出参数：
    - 'crown'： 嵌体
    - 'inner_dilation'： 嵌体内冠（有填充间隙）

### 返回内容

- 成功：{"Msg":{'data':out_items}, "Code": 200,"State": "Success", "version": "1.0.0"}
- 失败：{"error": error_message}

## 备注
