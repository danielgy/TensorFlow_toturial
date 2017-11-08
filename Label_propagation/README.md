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

其中![](http://latex.codecogs.com/gif.latex?\\p_{ij})表示从节点i转移到节点j的概率。
另外假设有M个分类N个样本，其中有label的为N1个，没有label的为N2个，则可以对label构建两个矩阵分别为N1*M的$y_l$和N2*M的$y_u$,$y_u$由于无标签信息，则可以随机初始化，此时定义![](http://latex.codecogs.com/gif.latex?\\f=[y_l,y_u ])。<br>
此时迭代计算过程如下：
> - 计算$f=pf$
> - 更高label的标签：$f_l=y_l$
> - 重复上述过程直至f收敛

**简化计算** <br>
如下所示，矩阵P构成为：
$$ P=\begin{bmatrix}
p_{ll} & p_{lu} \\ 
p_{ul} & p_{uu} 
\end{bmatrix} $$
由于$y_l$始终不变，因此实际计算的只有：
$$ f_{u}\leftarrow p_{uu}f_{u}+p_{ul}y_{l}$$
迭代上式至收敛则得到解。
另外，可以直接用如下公式得到解为：
$$ f_{u}=(I-p_{uu})^{-1}p_{ul}y_{l}$$


