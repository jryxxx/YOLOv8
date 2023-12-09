# YOLO v8

## 改进流程

- 编写模块算法，放在 ultralytics/nn 路径
- 更改 ultralytics/nn/tasks.py 文件，注册新模块;
  更改 parse_model 函数
- 更改 ultralytics/cfg/models/v8/yolov8.yaml 配置文件，更改网络结构

## 现有方法

- LSKA 模块
![Alt text](images/image.png)
- C2f 模块
![Alt text](images/image-1.png)
