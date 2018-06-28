---
layout: post
title:  "神经网络简单介绍(译)"
date:   2017-05-30 09:00:00 +0800
categories: [deeplearning,machine-learning]
---



<h2> 目录：</h2>

* [1. 神经网络简介](#1)
* [2. 多层感知机及其基本概念](#2)
* [3. 神经网络方法操作步骤](#3)
* [4. 图示神经网络计算过程](#4)
* [5. 后向算法数学推理](#5)
* [6. 参考文献](#6)
* [7. 附录](#7)


<h3 id="1">1. 神经网络简介 </h3>

在我们调试代码寻找bug的时候，我们通常的做法是在不同的环境或者输入不同的输入数据，通过对不同结果的分析，从而判断出bug应该会在哪个模块或者某行代码中出现。当你找到了出错的地方，并作出修改后，我们还是会使用不同的输入进行不断的测试，直到确定代码的正确性。

同理神经网络的运行逻辑也与调试bug类似。输入不同的样本数据，通过不同隐藏层(hidden layer)下不同的神经元(neurons)的计算，最后通过输出层(output layer)计算出最终的结果。这种预测结果的方法通常被称作 **前向传播(Forward Propagation)**。

接下来，我们将前向传播预测的结果与实际结果进行比较，这个过程最终的目的是让预测的结果尽量逼近实际结果。在这个过程中，每个神经元对最终输出结果都带来了一定的误差，那么我们该怎么减少这类误差？

类似寻找bug一样，我们可以采用向后推导，从而找出哪个神经元出现了偏差，修正这些神经元的权重，降低它们的权重，这个过程通常被称为 **后向传播(Backward Propagation)**。

为了更好更快的降低结果的误差，并且减少前向和后向的迭代次数。神经网络采用了 **梯度下降(“Gradient Descent)** 优化算法。

这就是神经网络的工作原理。通过这种简单直观的描述，主要是为了让大家对神经网络有一个初级的认识。


<h3 id="2">2. 多层感知机及其基本概念</h3>

就像地球上的物质是由原子构成，神经网络的基本单元就是 **感知机(perceptron)**。那么什么是感知机？

感知机可以被认为是通过获取多个输入，通过某种计算，最终得到一个输出的处理单元，如图1所示：

![感知机(perceptron)](/assets/images/neural_networks/perceptron.png)

通过上图可以看出，感知机获取三个输入，并最终输出一个结果。那么接下来的逻辑是，这三个输入与最终的输出结果之间是什么样的关系呢？让我们先从简单的逻辑开始，在此基础上再进一步引申出更复杂的逻辑。

下面，我们讨论了三种不同的输入输出关系：

1. **通过简单的组合逻辑将不同的输入组合在一起，并通过阈值计算出结果。** 例如： 假设 $$x_1=0, x_2=1, x_3=1$$ 并且同时设定阈值 $$\theta=0$$。因此能够得到，如果 $$x_1+x_2+x_3>0$$ 的输出结果为1, 否则结果为0。你可以看到，在这个例子中感知机的输出结果为1。

2. **接下来，我们对输入增加些权限。** 权重给不同的输入赋予了不同的重要性。例如：将 $$x_1, x_2, x_3$$ 的权重分别设置为 $$w_1=2, w_2=3, w_3=4$$ ， 为了计算输出结果，我们分别将输入数据与对应的权重相乘，并且与阈值相比较, 如公式所示：$$w_1\times x_1+w_2\times x_2+w_3\times x_3>\theta$$。 相比$$x_1, x_2, x_3$$ 获得了更高的权重。
3. **接下来，让我们增加些偏移量。** 每个感知机一般也会存在一些偏移量，这样可以保证感知机保持一定的灵活性(flexible)。 就像线性函数 $$y=ax+b$$, 它允许上下移动直线，从而让数据拟合的更合理。如果没有偏移量 $$b$$, 那么直线只会通过原点$$(0, 0)$$，这样得到的拟合结果就有可能不会太好。例如：当一个感知机有两个输入，那么它就需要要三个权重值，每个输入分别对应一个，偏移量对应一个。那么输入的线性表示就可以表示成：$$w_1\times x_1+w_2\times x_2+w_3\times x_3+1\times b$$

不过，上面描述的是一种简单的线性的感知机，看着比较简单也显得有点无趣。因此有人在此基础上对神经元的输入和偏移量增加了非线性变换(activation function)。

**什么是激活函数(activation function)**

激活函数是将输入与权重相乘得到的和值($$w_1\times x_1+w_2\times x_2+w_3\times x_3+1\times b$$) 作为输入参数，并最终算出神经元的输出结果。

在上面的公式中，我们可以将 $$1$$ 看做 $$x_0$$, $$b$$ 看做 $$w_0$$。

$$a = f(\sum w_ix_i)$$

激活函数更多用来做非线性变换，这样能让我们能够拟合更多的非线性的场景，或者估计更多负责的函数。目前有许多激活函数，例如：**Sigmoid**， **Tanh**， **Relu**等等。

**前向传播，后向传播，Epoch**

直到这里，我们知道了怎么计算神经网络的结果，这就是我们通常说的前向传播。但是如果计算得到的输出结果与实际的结果相差的比较多的情况下。在神经网络中会怎么处理这种问题呢，我们可以在错误的基础上不断的更新权重和偏移量的值。那么这个更新权重和偏移量的过程就是我们所说的后向传播(backward propagation)

后向传播根据最小化实际结果与预测结果的差距的原则下，沿着神经网络反向计算出每个神经元的权重。具体的数学原理我们会在后面的章节中介绍。

前向传播和后向传播这么一个迭代的过程我们称之为“Epoch”

**多层感知机**

现在让我们来看下什么是多层感知机。到目前为止，我们了解到了单层感知机拥有三个输入节点 $$x_1, x_2, x_3$$, 一个输出层。但是在实际的应用中单层神经网络的能力还是比较有限。多层感知机是在输入层(Input Layer)和输出层(Output Layer)中间添加多层隐含层(Hidden Layer), 如下图所示：

![多层感知机(perceptron)](/assets/images/neural_networks/multi_layer_perception.png)

上图展示的是包含一层隐藏层(使用绿色表示)，但是在实际应用中可以包含多层隐藏层。另外一个需要注意的地方是所有的层都是全连接，换句话说，每一层的所有节点都与前一层及下一层的所有节点两两相互连接。

现在我们对神经网络有了更进一步的了解，接下来让我们从如何训练神经网络（最小化误差）来深入了解神经网络。我们使用的是比较通用的训练算法 [梯度下降法](https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/)

**全量梯度下降(Full Batch Gradient Descent)和随机梯度下降(Stochastic Gradient Descent)**

这两种梯度下降法采用的都是相同的权重更新算法，唯一不同的地方主要在于使用多少训练样本。

正如全量梯度下降算法(Full Batch Gradient Descent)的名字所示，该算法每次都使用的是全量的训练数据来更新网络中的权重。相反随机梯度下降算法采用的一个或者更多训练样本，但绝对不是使用全量的训练样本来更新网络的权重。

让我们用一个简单的例子直观的感受下这两种算法的不同。假设我们有一个10个样本的训练集，两个权重值$$w_1, w_2$$ 

**Full Batch** 同时使用这10个样本来计算权重的改变量 $$w_1(\vartriangle{w_1})$$ 及 $$w_2(\vartriangle{w_2})$$, 并用这改变量来更新权重 $$w_1, w_2$$ 

**SGD** 首先使用第一个样本计算出权重的改变量 $$w_1(\vartriangle{w_1})$$ 及 $$w_2(\vartriangle{w_2})$$, 并且用这改变量来更新权重 $$w_1, w_2$$。然后再用第二个样本做同样的操作来更新权重。

大家可以通过阅读这篇文章来更进一步的了解[梯度下降法](https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/)


<h3 id="3">3. 神经网络方法操作步骤</h3>

![多层感知机(perceptron)](/assets/images/neural_networks/multi_layer_perception.png)

让我们来看下神经网络每一步都是怎么处理的。如上图所示，在输出层只有一个神经元，这个可以用来处理分类问题（预测0或者1）。我们同样可以使用两个神经元来同时预测两种类别。

让我们先看看整体的步骤：

1. 假定神经网络的输入及输出为：
	* $$X$$ 是一个输入矩阵
	* $$y$$ 是一个输出矩阵
2. 首先使用随机值来初始化权重和偏移量(这些权重和偏移量只在开始的时候初始化，后面就使用训练后更新得到的权重，偏移量), 我们通过下面的符号来表示：
	* $$wh$$ 表示隐藏层的权重
	* $$bh$$ 表示隐藏层的偏移量
	* $$wout$$ 表示输出层的权重
	* $$bout$$ 表示输出层的偏移量
3. 我们将输入矩阵和隐藏层权重矩阵点乘 加上隐藏层偏移矩阵得到的值作为隐藏层的输入，如下式所示：

	$$hidden\_layer\_input= matrix\_dot\_product(X,wh) + bh $$
	 
4. 使用激活函数(activation function) Sigmoid 做非线性变换。Sigmoid的输出结果样式为 $$1/(1+ exp(-x))$$

	$$hiddenlayer\_activations = sigmoid(hidden\_layer\_input)$$
	
5. 使用输出层的权重矩阵及偏移量矩阵对隐藏层激活函数的输出结果做线下变换，然后通过输出层的激活函数获得最终的输出结果，这里可以根据实际输出结果的样式选择合适的激活函数

	$$output\_layer\_input = matrix\_dot\_product (hiddenlayer\_activations * wout ) + bout$$
	
	$$output = sigmoid(output\_layer\_input)$$
	
**上面的五个步骤就是我们所说的 “前向传播”**

6. 前向传播计算得到的结果与实际结果进行比较，计算出它们之间的误差，通常这里可以使用最小均方误差表示 $$loss = ((Y-t)^2)/2$$

	 $$E = y – output$$
	 
7. 计算输出层，隐藏层神经元的梯度，我们通过计算每一层非线性激活函数导数值。Sigmoid函数求导得到的公式为：$$x*(1-x)$$

	$$slope\_output\_layer = derivatives\_sigmoid(output)$$
	
	$$slope\_hidden\_layer = derivatives\_sigmoid(hiddenlayer\_activations)$$
	
8. 计算输出层的改变量(delta)，采用输出误差与输出层激活函数的梯度

	$$d\_output = E * slope\_output\_layer$$
	
9. 在这一步，输出层的偏差就能反向传播回网络中，也就是我们所说的隐藏层的错误。我们会通过将第8步获取的改变量(delta)乘以隐藏层和输出层之间边的权重($$wout.T$$)

	$$Error\_at\_hidden\_layer = matrix\_dot\_product(d\_output, wout.Transpose)$$
	
10. 计算出隐藏层的偏差, 将隐藏层的偏差乘以隐藏层激活函数的导数

	$$d\_hiddenlayer = Error\_at\_hidden\_layer * slope\_hidden\_layer$$
	
11. 更新隐藏层与输出层线性函数的权重

	$$wout = wout + matrix\_dot\_product(hiddenlayer\_activations.Transpose, d\_output)*learning\_rate$$
	
	$$wh =  wh + matrix\_dot\_product(X.Transpose,d\_hiddenlayer)*learning\_rate$$
	
	**learing_rate:权重更新步长** 
	
12. 更新隐藏层与输出层线性函数的偏移量

	$$bh = bh + sum(d\_hiddenlayer, axis=0) * learning\_rate$$
	$$bout = bout + sum(d\_output, axis=0)*learning\_rate$$
	
**第六步到第十二步就是我们所说的后向传播**

一个前向传播加上一个后向传播组成了一次训练迭代过程。正如我们之前所说的，我们使用更新后的权重和偏移量在进行第二次的迭代。

我们上面描述的是采用Full Batch Gradient Descent来更新权重及偏移量

<h3 id="4"> 4.图示神经网络计算过程 </h3>

我们将展示上述神经网络前向传播，后向传播所使用的输入矩阵，权重矩阵，偏移量矩阵，错误矩阵是怎么变化的，从而让大家能更加直观的感受多层感知机神经网络内部计算逻辑。

**注：**
 * 为了更好的展示效果，小数都保留两到三位小数
 * 黄色区域表示当前激活的区域
 * 橙色区域表示用来计算黄色区域值得输入

**Step 0:** 读取输入和输出矩阵

![step 0](/assets/images/neural_networks/neural_network_visualization_step0.jpg)

**Step 1** 初始化权重矩阵和偏移矩阵(初始化权重矩阵和偏移矩阵的方法有很多，这边采用的随机方法)

![step 1](/assets/images/neural_networks/neural_network_visualization_step1.png)

**Step 2** 计算隐藏层的输入

$$hidden\_layer\_input= matrix\_dot\_product(X,wh) + bh$$

![step 2](/assets/images/neural_networks/neural_network_visualization_step2.png)

**Step 3** 对隐藏层做非线性变换

$$hiddenlayer\_activations = sigmoid(hidden\_layer\_input)$$

![step 3](/assets/images/neural_networks/neural_network_visualization_step3.png)

**Step 4** 计算输出层线性和非线性变换的结果

$$output\_layer\_input = matrix\_dot\_product (hiddenlayer\_activations * wout ) + bout$$

$$output = sigmoid(output\_layer\_input)$$

![step 4](/assets/images/neural_networks/neural_network_visualization_step4.png)

**Step 5** 计算预测误差

$$E = y-output$$

![step 5](/assets/images/neural_networks/neural_network_visualization_step5.png)

**Step 6** 计算输出层，隐藏层梯度

$$Slope\_output\_layer= derivatives\_sigmoid(output)$$

$$Slope\_hidden\_layer = derivatives\_sigmoid(hiddenlayer\_activations)$$

![step 6](/assets/images/neural_networks/neural_network_visualization_step6.png)

**Step 7** 计算输出层增量

$$d\_output = E * slope\_output\_layer*lr$$

![step 7](/assets/images/neural_networks/neural_network_visualization_step7.png)

**Step 8** 计算隐藏层的偏差

$$Error\_at\_hidden\_layer = matrix\_dot\_product(d\_output, wout.Transpose)$$

![step 8](/assets/images/neural_networks/neural_network_visualization_step8.png)

**Step 9** 计算隐藏层的增量

$$d\_hiddenlayer = Error\_at\_hidden\_layer * slope\_hidden\_layer$$

![step 9](/assets/images/neural_networks/neural_network_visualization_step9.png)

**Step 10** 更新输出层，隐藏层权重

$$wout = wout + matrix\_dot\_product(hiddenlayer\_activations.Transpose, d_output)*learning\_rate$$

$$wh =  wh+ matrix\_dot\_product(X.Transpose,d\_hiddenlaye$$

![step 10](/assets/images/neural_networks/neural_network_visualization_step10.png)


**Step 11** 更新输出层，隐藏层偏移量

$$bh = bh + sum(d\_hiddenlayer, axis=0) * learning
\_rate$$

$$bout = bout + sum(d\_output, axis=0)*learning\_rate$$

![step 11](/assets/images/neural_networks/neural_network_visualization_step11.png)


<h3 id="5">5. 后向算法数学推理</h3>

我们先假设:

 * $$W_i$$ 为输入层与隐藏层之间的权重
 * $$W_h$$ 为隐藏层与输出层之间的权重

那么，$$h=\sigma(U)=\sigma(W_iX)$$, $$h$$ 是 $$U$$的函数，$$U$$ 是 $$W_i$$ 和 $$X$$ 的函数。 在这里我们使用$$\sigma$$ 表示函数。

$$Y=\sigma(U')=\sigma(W_hh)$$, $$Y$$ 是 $$U'$$ 的函数，$$U'$$ 是 $$W_h$$ 和 $$h$$ 的函数。
 
我们根据上面定义的公式来计算相应的函数的偏导数

后向传播主要计算两项偏导数 $$\partial E/\partial W_i$$ 和 $$\partial E/\partial W_h$$ 。就是我们通常说的输入层与隐藏层之间权重改变下错误的改变， 隐藏层与输出层之间权重改变下错误的改变。

为了计算上述的两个偏导数，我们需要使用链式求导法，因为 $$E$$ 是 $$Y$$ 的函数，$$Y$$ 是 $$U'$$ 的函数，$$U'$$ 是$$W_i$$的函数。

让我们很好的使用这个属性并计算梯度。

$$\partial E/\partial W_h = (\partial E/\partial Y)(\partial Y/\partial U')(\partial U'/\partial W_h)....$$  (1)

我们知道最小均方误差函数形式为：$$E = (Y-t)^2/2$$ 所以：

$$\partial E/\partial Y = (Y - t)$$

然而 $$\sigma$$ 是一个Sigmoid函数，Sigmoid函数的导数有一种非常有趣的表示方式就是 $$\sigma(1-\sigma)$$, 那么我们可以得到:

$$\partial Y/\partial U' = \partial(\sigma(U'))/\partial U‘ = \sigma(U')(1-\sigma(U'))$$

同时 $$Y=\sigma(U')$$, 那么：

$$\partial Y/\partial U' = Y(1-Y)$$

最后，$$(\partial U'/\partial W_h)=\partial(W_hh)/\partial W_h = h$$

将上述结果带入公式（1）中得到：

$$\partial E/\partial W_h = (Y-t)Y(1-Y)h$$

到目前为止，我们已经计算了隐藏层到输出层之间的梯度。那接下来我们可以计算隐藏层到输入层之间的梯度了：

$$\partial E/\partial W_i = (\partial E/\partial h)(\partial h/\partial U)(\partial U/\partial W_i)$$

同时，$$\partial E/\partial h = (\partial E/\partial Y)(\partial Y/\partial U')(\partial U'/\partial h)$$ 将该公式带入上式，则可以得到：

$$\partial E/\partial W_i = [(\partial E/\partial Y)(\partial Y/\partial U')(\partial U'/\partial h)](\partial h/\partial U)(\partial U/\partial W_i)......$$  (2)

所以，通过公式(2)大家可以看出先计算输出层与隐藏层之间的偏导数的优势了吗？

正如我们所看到的，在公式(2)中我们已经计算了 $$\partial E/\partial Y$$ 和 $$\partial Y/\partial U'$$ ,这样为我们减少了计算的时间和存储的空间。 这我们就可以知道为什么该算法叫做后向传播算法了。

让我们计算公式(2)中未计算的偏导数：

$$\partial U'/\partial h = \partial(W_hh)/\partial h = W_h$$

$$\partial h/\partial U = \partial(\sigma(U))/\partial U = \sigma(U)(1-\sigma(U))$$

然而，$$\sigma(U) = h$$, 所以：

$$\partial Y/\partial U = h(1-h)$$

其中， $$\partial U/\partial W_i = \partial(W_iX)/\partial W_i = X$$ 将上述公式代入公式(2)，我们能得到：

$$\partial E/\partial W_i = [(Y-t)Y(1-Y)W_h]h(1-h)X$$

所以，现在我们已经计算了所有层之间的梯度，那么权重可以通过下面的方式进行更新了：

$$W_h = W_h + \eta *\partial E/\partial W_h$$

$$W_i = W_i + \eta *\partial E/\partial W_i$$

其中，$$\eta$$ 是学习率(learning rate)

所以回头看看，为什么该算法叫做后向传播算法？ 主要原因我们可以看到：在 $$\partial E/\partial W_h$$ 和 $$\partial E/\partial W_i$$ 中，我们都能看到输出误差$$(Y-t)$$，这就是整个算法的起点，并且反向传播回到输入层来进行权重的更新。

那么，这些公式在代码中对应的代码是什么呢？

$$hiddenlayer\_activations=h$$

$$E= Y-t$$

$$Slope\_output\_layer = Y(1-Y)$$

$$lr = \eta$$

$$slope\_hidden\_layer = h(1-h)$$

$$wout = W_h$$

接下来你可以很容易的将数学公式与代码对应起来了。


<h3 id="6">6. 参考文献 </h3>

* [Understanding and coding Neural Networks From Scratch in Python and R](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/)

<h3 id="7">7. 附录</h3>

* [Python Numpy实现前向后向传播](https://github.com/everestocean/Algorithm/blob/master/machine_learning/deep_learning/deep_learning/neural_network.py)

