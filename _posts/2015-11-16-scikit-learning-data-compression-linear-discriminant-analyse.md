---
layout: post
title:  "scikit-learn: 通过降维方法对数据进行压缩2-线性判别分析(LDA)"
date:   2015-11-16 09:00:00 +0800
categories: [machine-learning, scikit-learn]
---

### 线性判别分析(Linear Discriminant Analysis LDA)

LDA则更多的是考虑了分类标签信息，寻求投影后不同类别之间数据点距离更大化以及同一类别数据点距离最小化，即选择分类性能最好的方向。PCA主要是从特征的协方差角度，去找到比较好的投影方式，即选择样本点投影具有最大方差的方向。
