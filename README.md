# pytorch-deeplabv3-training
用你的数据集训练DeepLabV3

请根据https://github.com/bubbliiiing/deeplabv3-plus-pytorch/ 制作数据集 并放入VOCdevkit文件夹

根据数据集 修改train.py里面的设置

运行`python train.py`

导出ONNX（Labview ONNX工具包推理使用的格式）请查看export.ipynb
_______________
TODO
+增加Loss计算种类支持
+增加推理代码

感谢 [bubbliiiing](https://github.com/bubbliiiing) | [msminhas93](https://github.com/msminhas93)
