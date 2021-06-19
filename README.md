# pbtree
Distribution prediction model by Probability Boosting TREE method. This framework can be used to predict various types of distributions for target variables, currently normal and gamma distributions are mainly supported.

# 概率提升树

目前主流的机器学习工具一般都是基于点估计，即只估计目标分布的均值，例如xgboost、lightgbm、LR等。这在大多数场景下是适用的，但是在某些场景下（例如小样本，或者需要强解释），我们不仅希望预估均值，还希望预估置信区间。当前的

我们希望开发一款具有以下特点的工具：

- 支持预估目标的全部分布参数。
- 支持工业界的应用：支持多线程训练、预估，支持在线服务。
- 支持特征交叉。
- 具有较好的解释性。
![pbtree](./image/pbtree3_1.mdl.png)