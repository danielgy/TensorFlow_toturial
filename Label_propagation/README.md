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
![fraction1](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction1.png?raw=true)
其中`$\alpha$`为超参数。（另外一种构建图的方式是knn图，即只保留每个节点的k近权重，其他边为0，此时的边权重为稀疏的。）<br>
**标签传播** <br>
标签传播即通过节点间的边传播标签信息，边的权重越大，表示两个节点越相似，此时的概率转移矩阵为：<br>
![fraction2](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction2.gif?raw=true) 
