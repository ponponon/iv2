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
