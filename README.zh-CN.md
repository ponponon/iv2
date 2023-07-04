# iv2

[简体中文](./README.zh-CN.md) | [English](./README.md)

## 简介

将图片转成向量，可用于以图搜图和图片相似度比较

> 向量纬度为 512

## 安装

Python 解释器:

- CPython : 3.8 及以上版本

安装方式:

```shell
pip install iv2
```

## 使用示例

```python
from pathlib import Path
from typing import List
from iv2 import ResNet, l2
from iv2.model import ResNet47_50Net


# Initialize a residual neural network
resnet: ResNet = ResNet(
    runtime_model='models/gl18-tl-resnet50-gem-w-83fdc30.pth',
    device='cpu'
)


vector_1: List[float] = resnet.gen_vector('p1.png')
vector_2: List[float] = resnet.gen_vector('p.png')

print(l2(vector_1, vector_2, sqrt=False))
```

## 参考项目

- [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
