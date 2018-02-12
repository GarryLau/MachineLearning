http://cv-tricks.com/object-detection/single-shot-multibox-detector-ssd/

深度学习相关的目标检测方法也可以大致分为两派：
需要region proposal的，如R-CNN、SPP-net、Fast R-CNN、Faster R-CNN、R-FCN；
端到端（End-to-End）无需区域提名的，如YOLO、SSD。

SSD是在精度和效率方面都较为不错的检测算法。
SSD中检测问题是转换为分类问题来做的。分类问题往往是预测出图片中物体的标签值；检测问题不仅仅要预测出物体的标签值还要找出物体的bounding box。
分类问题中物体往往占据图片的很大比例；而检测问题往往是一幅图片中有多个不同大小的物体，不仅需要预测出各个物体的标签值还需要找出各个物体的bounding box。
简而言之检测问题需要给出物体的bounding box和label，因此检测网络的输出需要有：
1. 类别概率；
2. bounding box坐标，可以用bounding box的中心点和宽高来表示，（cx,cy,h,w）。
需要注意的是在检测问题中类别需要加入背景类别，因为图片中很多区域是没有物体的。

为说明问题方便假设数据集只有猫狗，因此类别标签总共有三种，使用one-hot编码表示，[1 0 0]代表猫，[0 1 0]代表狗，[0 0 1]代表背景。

