---
title: 图像特征点
date: 2024-03-07 15:04:23
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: complete
destination: 03-Projects/01-Note/01-SLAM
share: true
obsidianUIMode: source
number headings: auto, first-level 1, max 6, _.1.1
---

## 1 BoW
ORB_SLAM 使用 BoW 进行相似图像检索，在重定位、回环检测、参考关键帧跟踪过程中都有用到。BoW 是一种 K 叉树形式的分类树，在构造 K 叉树时，根据特征描述的相似性作为距离度量进行聚类，树的叶子节点被计为单词，每个单词使用特征点的聚类中心表示。在进行特征匹配时，根据特征点是否落在同一个叶子节点判断是否对应同一个单词，同时可以得到一个相似度得分。一般在进行图像特征匹配时会根据公共单词数量和最小的相似度得分判断图像之间的相似性。
![1-Image_Match.png](1-Image_Match.png)
### 1.1 TF-IDF
TF（Term Frequency），指在单帧中某个单词的频次，频次高，权重大
IDF（Inverse Document Frequency），某个单词在词典中出现的频次，频次越低，则辨识度越高，相应权重 IDF 越大
最终 BoW 的权重是 `TF*IDF` ，每个单词都有自己的权重。

### 1.2 正向索引
再计算BoW时，每帧图像会记录一个正向索引表，记录在kd树某一层，命中的节点集合，以及节点中的特征点，用于加速两帧图像之间的特征点匹配。
### 1.3 逆向索引
orb-slam 中维护了一个关键帧数据库，其中有一个单词的逆向索引表，其记录了包含这个单词的关键帧和权重，用于快速查找匹配关键帧。

### 1.4 相似性对量
1. L1  ，各维度的差值的绝对值求和
2. 余弦相似度 ，向量夹角

### 1.5 levelsup
ORB_SLAM 在对图像生成词向量时，会根据 levelsup 记录图像中的单词落在了某层的哪些分支，这样在进行单词匹配时，只需要查找对应分支而不是从根节点开始查找，提高查找效率。
![2-Image_Match.png](2-Image_Match.png)


Reference:
[基于词袋模型的图像匹配 - Line's Blog](https://xhy3054.github.io/2019/04/19/2019-04-19-bow/)