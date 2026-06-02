# Grad-CAM 严格版可视化说明

## 升级内容

原有的 `visualize_pair_similarity.py` 采用的是基于 patch token 两两相似度的热力图，可用于定性展示，但严格来说并不是神经网络可解释性意义上的原生注意力图。

本次升级后，脚本新增了基于梯度的 `Grad-CAM` 可视化模式：

- 脚本路径：[`visualize_pair_similarity.py`](D:/workspace/图纸检索项目/数据集/111_test/experiments/visualize_pair_similarity.py)
- 默认方法：`grad_cam`
- 备选方法：`patch_similarity`

## 方法说明

1. 对查询图和候选图分别经过 CLIP 视觉编码器与检索适配器，得到检索嵌入。
2. 以两幅图的嵌入余弦相似度作为目标分数。
3. 反向传播该相似度分数到视觉 patch token。
4. 根据 token 梯度的通道均值计算各 patch 的 Grad-CAM 强度。
5. 将得到的热力图重新映射回图像空间，用于展示模型在“判定两图相似”时最关注的区域。

## 输出文件

每次运行会生成两类文件：

1. `PNG` 可视化图：展示查询图、候选图及其 Grad-CAM 热力图。
2. `JSON` 分析报告：包含语义相似度、结构相似度、文本相似度、SSIM、Grad-CAM 对应的成对分数等信息。

## 论文可用表述

为提高可解释性分析的严谨性，本文进一步采用基于梯度的 Grad-CAM 方法对检索模型进行可视化。具体而言，以查询图与候选图的嵌入相似度作为目标分数，反向传播至视觉 patch token，并据此生成对应的热力响应图。相比于简单的 patch 相似性热力图，Grad-CAM 更能够反映模型在判定两图相似时实际依赖的关键区域，因此更适合作为论文中的定性分析工具。
