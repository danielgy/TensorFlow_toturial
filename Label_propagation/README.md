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
![fraction1](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction1.gif?raw=true) <br>
其中![](http://latex.codecogs.com/gif.latex?\\alpha)为超参数。（另外一种构建图的方式是knn图，即只保留每个节点的k近权重，其他边为0，此时的边权重为稀疏的。）<br>
**标签传播** <br>
标签传播即通过节点间的边传播标签信息，边的权重越大，表示两个节点越相似，此时的概率转移矩阵为：<br>
![fraction2](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction2.gif?raw=true) <br>

其中![](http://latex.codecogs.com/gif.latex?\p_{ij})表示从节点i转移到节点j的概率。
另外假设有M个分类N个样本，其中有label的为N1个，没有label的为N2个，则可以对label构建两个矩阵分别为N1*M的![](http://latex.codecogs.com/gif.latex?\\y_l)和N2*M的![](http://latex.codecogs.com/gif.latex?\\y_u),![](http://latex.codecogs.com/gif.latex?\\y_u)由于无标签信息，则可以随机初始化，此时定义![](http://latex.codecogs.com/gif.latex?\\f=[y_l,y_u ])。<br>
此时迭代计算过程如下：
> - 计算f=pf
> - 更新label的标签：![](http://latex.codecogs.com/gif.latex?\\f_l=y_l)
> - 重复上述过程直至f收敛

**简化计算** <br>
如下所示，矩阵P构成为：
![fraction3](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction3.gif?raw=true) <br>
由于![](http://latex.codecogs.com/gif.latex?\\y_l)始终不变，因此实际计算的只有：
![fraction4](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction4.gif?raw=true) <br>
迭代上式至收敛则得到解。
另外，可以直接用如下公式得到解为：
![fraction5](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/fraction5.gif?raw=true) <br>
### 实验结果： <br>

![result1](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result1.png?raw=true) <br>
![result2](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result2.png?raw=true) <br>
![result3](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result3.png?raw=true) <br>
![result4](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result4.png?raw=true) <br>
![result5](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result5.png?raw=true) <br>
![result6](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result6.png?raw=true) <br>
![result7](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result7.png?raw=true) <br>
![result8](https://github.com/danielgy/TensorFlow_toturial/blob/master/Label_propagation/images/result8.png?raw=true) <br>
