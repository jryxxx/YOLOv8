# YOLO v8

## 改进流程

- 编写模块算法，放在 ultralytics/nn 路径
- 更改 ultralytics/nn/tasks.py 文件，注册新模块 \\
  更改 parse_model 函数
- 更改配置文件，放在 ultralytics/cfg/models/v8 路径

## 现有方法

- LSKA 模块
  - 算法实现 attention/LSKA.py
  - 配置文件 cfg/LSKA.yaml
  - ultralytics/nn/tasks.py 修改

  ```python
  from ultralytics.nn.attention.LSKA import C2f_LSKA_Attention
  def parse_model(d, ch, verbose=True):
    # 加入 C2f_LSKA_Attention
    if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, C2f_LSKA_Attention):
    # 加入 C2f_LSKA_Attention
    if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost,C3x,C2f_LSKA_Attention):
                args.insert(2, n)
                n = 1
  ```

  - 实验结果 ![Alt text](results/n/resLSKA2.png)
- CoordAttention 模块
  - 算法实现 attention/CoordAttention.py
  - 配置文件 cfg/CoordAttention.yaml
  - ultralytics/nn/tasks.py 修改

  ```python
  from ultralytics.nn.attention.CoordAttention import CoordAtt
  def parse_model(d, ch, verbose=True):
    # 加入 CoordAtt
    if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, CoordAtt):
  ```

  - 实验结果 ![Alt text](results/n/resCoordAttention2.png)
