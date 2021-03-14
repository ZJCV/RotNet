<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/RotNet"><img align="center" src="./imgs/RotNet.png"></a></div>

<p align="center">
  «ZCls»实现了基于深度学习的图像旋转校正 
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
  <a href="https://pypi.org/project/zcls/"><img src="https://img.shields.io/badge/PYPI-zcls-brightgreen"></a>
</p>

* 模型：`MobileNetV2`
* 优化器：`SGD`
* 学习率调度器：`MultiStepLR`
* 批量大小：`128`
* 样本转换：缩放+颜色抖动+灰度+归一化+随机擦除+随机旋转（*随机边界值填充*）
* 数据集：[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)

使用`RTX 2080Ti`共训练`2`万轮，得到最高的测试集精度为`98.7%`，平均单次推导时间为`8ms`

![](./imgs/demo.png)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [使用](#使用)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

在网上找资料，发现可以通过深度学习算法检测图像旋转角度，参考：

* [d4nst/RotNet](https://github.com/d4nst/RotNet)
* [Correcting Image Orientation Using Convolutional Neural Networks](https://d4nst.github.io/2017/01/12/image-orientation/)
* [Image Orientation Estimation with Convolutional Networks](https://lmb.informatik.uni-freiburg.de/Publications/2015/FDB15/image_orientation.pdf)
* [UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS](https://arxiv.org/pdf/1803.07728.pdf)

其相应的实现并不能满足当前的性能要求，所以自己实现一个

## 使用

* 训练

```
$ export PYTHONPATH=<仓库根路径>
$ CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg=configs/xxx.yaml
```

* 测试

```
$ export PYTHONPATH=<仓库根路径>
$ CUDA_VISIBLE_DEVICES=0 python demo/demo.py -cfg=demo/xxx.yaml
```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [d4nst/RotNet](https://github.com/d4nst/RotNet)
* [ZJCV/ZCls](https://github.com/ZJCV/ZCls)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/RotNet/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjykzj