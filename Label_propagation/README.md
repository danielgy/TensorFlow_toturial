# label propagation(标签传播)

## label propagation 原理

label propagation是一种半监督学习算法，主要基于三种假设：<br>
- 平滑假设： 相似的数据具有相同的label
- 聚类假设： 处于同一聚类下的数据具有相同的label
- 流形假设： 处于同一流行结构下的数据具有相同的label

### 步骤： <br>
1.构建相似矩阵 <br> 2. 标签传播 <br>

**构建相似矩阵** <br>
标签传播算法是基于图模型的，第一步我们需要先构建一个图，图中的每个节点都是一个数据，边表示数据之间的相似度，假设我们所构建的图为全连接图，定义节点i和节点j的相似度（即边权重）为：<br>
$$ w_ij =exp(-\dfrac{||x_i-x_j||}{a^2}) $$



