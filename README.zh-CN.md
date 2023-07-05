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

运行模型，需要权重文件，

下载地址: http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/

下载该地址中的 `gl18-tl-resnet50-gem-w-83fdc30.pth` 文件

## 使用示例

```python
from typing import List
from iv2 import ResNet, l2


# Initialize a residual neural network
# download: http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/
resnet: ResNet = ResNet(
    runtime_model='gl18-tl-resnet50-gem-w-83fdc30.pth',
    device='cpu'
)


vector_1: List[float] = resnet.gen_vector('p1.png')
vector_2: List[float] = resnet.gen_vector('p.png')

print(l2(vector_1, vector_2, sqrt=False))
```

## 参考项目

- [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
