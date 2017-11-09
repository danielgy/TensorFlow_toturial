# 混合专家系统（Mixture of Experts）

### **原理：**
混合专家系统（MoE）是一种神经网络，也属于一种combine的模型。适用于数据集中的数据产生方式不同。不同于一般的神经网络的是它根据数据进行分离训练多个模型，各个模型被称为**专家**，而**门控模块**用于选择使用哪个专家，模型的实际输出为各个模型的输出与门控模型的权重组合。各个专家模型可采用不同的函数（各种线性或非线性函数）。混合专家系统就是将多个模型整合到一个单独的任务中。 <br>

混合专家系统有两种架构：competitive MoE 和cooperative MoE。competitive MoE中数据的局部区域被强制集中在数据的各离散空间，而cooperative MoE没有进行强制限制。<br>

对于较小的数据集，该模型的表现可能不太好，但随着数据集规模的增大，该模型的表现会有明显的提高。

定义X为N*d维输入，y为N*c维输出,K为专家数，![](http://latex.codecogs.com/gif.latex?\\lambda) 为学习率：<br>
![fraction1](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_1.gif?raw=true) <br>
 各专家输出为:   ![fraction2](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_2.gif?raw=true) <br>
*(其中![](http://latex.codecogs.com/gif.latex?\w_{ik})为第k个专家模型对第i列输出的权重，![](http://latex.codecogs.com/gif.latex?\V_{ik})为第k个专家对第i列的预测。（$w_{ik}$添加了bias所以输出为d+1维）)*

第k个专家输出均值为：![fraction3](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_3.gif?raw=true) <br>

门限模块输出为：![fraction4](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_4.gif?raw=true) <br>
输出$y_i$通过softmax函数转成概率值为：![fraction5](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_5.gif?raw=true) <br>

对于Cooperative MoE：<br>
![fraction6](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_6.gif?raw=true) <br>
![fraction7](https://github.com/ZoeYuhan/machine-learning/blob/master/images/moe_7.gif?raw=true) <br>

对于Competitive MoE：<br>
![fraction8](https://github.com/ZoeYuhan/machine-learning/blob/master/images/moe_8.gif?raw=true) <br>
![fraction9](https://github.com/ZoeYuhan/machine-learning/blob/master/images/MOE_9.gif?raw=true) <br>
![fraction10](https://github.com/ZoeYuhan/machine-learning/blob/master/images/moe_10.gif?raw=true) <br>
![fraction11](https://github.com/ZoeYuhan/machine-learning/blob/master/images/moe_11.gif?raw=true) <br>


### **实验结果：**

#### **不同数据集相同k值：**
1. k=2使用线性数据集，采用SGD和FTRL两种训练方式，结果如下：
![bar1](https://github.com/ZoeYuhan/machine-learning/blob/master/images/bat_result1.png?raw=true) <br>
![bar2](https://github.com/ZoeYuhan/machine-learning/blob/master/images/bat_result2.png?raw=true) <br>
2. k=2使用非线性数据集，采用SGD和FTRL两种训练方式，结果如下：
![circle1](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_result1.png?raw=true) <br>
![circle2](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_result2.png?raw=true) <br>


#### **相同数据集不同k值：**
1. k=1:
![k=1](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=1_2.png?raw=true) <br>
![k=1](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=1_1.png?raw=true) <br>
2. k=2:
![k=2](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=2_1.png?raw=true) <br>
![k=2](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=2_2.png?raw=true) <br>
3. k=4:
![k=4](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=4_1.png?raw=true) <br>
![k=4](https://github.com/ZoeYuhan/machine-learning/blob/master/images/circle_k=4_2.png?raw=true) <br>
