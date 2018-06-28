---
layout: post
title:  "生成对抗网络简介(译)"
date:   2017-06-17 09:00:00 +0800
categories: [deeplearning, machine-learning]
---

## 简介

神经网络在过去的纪念得到了长足的发展。从而帮助机器在图像和语音识别上也能够达到与人类相当的水平。并且在自然语言的理解上也达到了一定的水准。

但即使如此，要让机器完全取代人类实现自动化的任务处理就显得有点牵强了。毕竟我们所做的不仅仅是识别图像中的物体或者或者理解周围的人说的是什么--对不对？

让我们看下在哪些领域是需要人类独特创造力的。

* 训练一个人工智能作者，通过学习Analytics Vidhya中的历史文章，使得机器能够简单明了的写出一篇描述数据科学概念的文字
* 对于普罗大众想要购买一幅知名画家的画作是一件不容易的事情，因为它们都比较的贵。那么我们能够训练出一个人工智能画家，通过学习某一知名画家过去的作品，使得机器能够独立自主的画出与知名画家相类似的画作。

你认为，这些任务可以通过机器来完成？那么，答案也许会使你感到惊讶。

当然对这些任务的自动化处理确实是非常困难的，但是生成对抗网络(GAN) 已经开始使得这些任务变成有可能完成的。

如果你被生成对抗网络(GAN) 这个名字吓到，不用担心，等到你阅读完这篇文章后你也许就不会感到害怕了。

在这篇文章中，将会为大家介绍对抗神经网络(GAN)相关概念，以及它是如何工作的。同时也会介绍一些大家使用GAN做的有趣的事情，同时提供相关学习资源让大家足以进一步的深入学习。

## 目录
* [1. 什么是生成对抗网络(GAN](#1)
* [2. 生成对抗网络如何工作](#2)
* [3. 生成对抗网络面对的挑战](#3)
* [4. 生成对抗网络的应用](#4)
* [5. 资源](#5)
* [6. 参考文献](#6)
* [7. 附录](#7)


<h3 id="1">什么是生成对抗网络(GAN)</h3>

Yann LeCun, 深度学习领域的一位知名人士在Quota中的表示：

**“(GANs), and the variations that are now being proposed is the most interesting idea in the last 10 years in ML, in my opinion.”**

他的观点确实有道理。

但是到底什么是生成对抗网络(GAN)

让我们简单的分析下生成对抗网络的概念：

如果你想在某一领域做的更好，比如说下棋，那么你会怎么做呢？也许你会找个比你更厉害的人，跟他一起对弈。然后你会分析自己哪做错了，哪些地方他\她做的比你更好，紧接着会思考以后该怎么做能够在以后的对弈中做的比他\她更好。

你会不断的重复这样的步骤，直到最终打败对方。这个概念也同样可以被利用来构建更好的模型。简而言之，为了获得一个更强大的英雄(生成器Generator)，我们需要一个更强大的对手(分辨器Discriminator)。

**根据现实生活实例的另一种分析**

一个比较实际的例子就是伪造者和艺术作品鉴定员之间的关系。

<div align="center">
<img src="/assets/images/gan/forge.jpg" width="60%" height="60%"  />
</div>

伪造者主要的工作是通过模仿某一知名艺术家的手法伪造出一幅新的作品。如果这幅作品足以以假乱真，那么伪造者就能从中收获一笔丰厚的收益。

另一个角度，一个艺术作品鉴定员的任务就是找出这些伪造者创作的作品，并且将伪造者绳之于法。那么他需要怎么做才能达到目标呢？他要对原作具有相当的了解，同时知道对应艺术家的创作手法。并且通过不断的学习来提升自己的鉴别能力。

这场伪造者和鉴定者之间的较量会不断的继续，最终训练出世界级的鉴别师。这是一场善与恶之间的较量。

<h3 id="2">对抗神经网络是如何工作的</h3>

到这我们队生成对抗网络(GAN) 有了整体上的了解。那么接下来，我们来了解下生成对抗网络的算法细节。

如下图所示，生成对抗网络(GAN)中包含两部分--生成神经网络(Generator Neural Network)和判别神经网络(Discriminator Neural Network)

<div align="center">
<img src="/assets/images/gan/g1.jpg" width="70%" height="70%"  />
</div>

其中生成网络接收一个随机输入同时通过相关算法生成一个样本数据。如上图所示，我们可以看到生成器($$G(z)$$)接收一个符合分布$$p(z)$$的输入$$z$$，其中$$z$$是根据概率分布$$p(z)$$获得的样本。然后通过生成器生成一数据并传入判别网络$$D(x)$$中。判别网络一面接收真实数据的输入，另一方面接收生成器生成的数据，同时判断输入的数据是真实数据还是生成的数据。判别网络接收一个输入样本$$x$$来自真实数据的分布函数$$p_{data}(x)$$。然后判断网络$$D(x)$$作二元判断，通过使用sigmod函数得到输出 0或者1。

让我们定义一些符号，方便后面进一步分析生成对抗网络(GAN)

$$P_{data}(x)$$ --> 真实数据的分布

$$X$$ --> 样本数据服从真实数据分布$$P_{data}(x)$$

$$P(z)$$ --> 生成器的分布

$$Z$$ --> 来自生成器分布$$P(z)$$的样本

$$G(z)$$ --> 生成器网络

$$D(x)$$ --> 判别器网络

接着生成对抗网络的训练通过生成器和判别器之间的博弈到收敛后结束。这可以通过以下公式进行描述：

$$min_{G}max_{D}V(D, G)$$

$$V(D, G) = E_{x\thicksim p_{data}(x)}[logD(x)] + E_{z\thicksim p_{z}(z)}[log(1-D(G(z)))]$$

在函数$$V(D, G)$$中，第一项是来自实际分布($$p_{data}(x)$$)的数据，在通过判别器时得到的熵。判别器希望想办法最大化该值使其更接近1。第二项是一个符合分布p(z)的随机数据，通过生成器生成一个伪数据，并且通过判别器进行判断是否为假数据，对于这项判别器努力最大化它使他更接近0。所以最终判别器希望最大化函数$$V(D, G)$$。

相反，生成器网络的主要任务是最小化函数$$V(D, G)$$。从而减少真实数据和伪造数据之间的差异。换句话说生成器和判别器之间的关系就好比猫和老鼠的关系。

**生成对抗网络训练部分**

广泛而言，训练阶段分为两个主要部分，并且他们之间是按照顺序进行的操作。

* __Part1:__ 训练判别器网络并且冻结生成器网络(冻结主要的意思是将训练设置为false，这时候只处理前向传播不进行后向传播)

	<div align="center">
	<img src="/assets/images/gan/s1.jpg" width="60%" height="60%"  />
	</div>


* __Part2:__ 训练生成器网络冻结判别器网络

	<div align="center">
	<img src="/assets/images/gan/s21.png" width="60%" height="60%"  />
	</div>


**训练GAN的步骤**

* __Step1: 问题定义__ 你是想生成一幅伪图片还是一篇伪文章，你需要先定义好你需要处理的问题，并不断的搜集相关数据
* __Step2: 定义GAN的结构__ 定义你的GAN的结构，对应生成器网络和判别器网络都是多层感知机，或者是卷积神经网络？这一步定义完全依赖你需要处理的问题。
* __Step3: 根据真实数据训练判别器网络__ 使用你想伪造的真实数据来训练判别器网络，使得它有足够的能力来正确判断真实数据。
* __Step4: 为生成器生成一些假数据并且训练判别器来识别伪数据__ 获得生成器生成的数据，并使得判别器能够判断数据是伪造的。
* __Step5: 使用判别网络的输出来训练生成器网络__ 在训练完判别器网络后，我们就能够使用判别器的结果来知道生成器的训练。训练生成器来足以欺骗判别器网络
* __Step6: 多次重复3到5步__ 
* __Step7: 人工判断伪造的数据是否足够真实，如果是，那么可以停止训练，否则继续重复3到5步__ 这一步是手动操作，人工判断构造出来的伪数据是否足以以假乱真。当这步完成后你可以判断GAN网络是否表现的够好

让我们想象下，如果你有个功能齐全的生成器网络，你就能复制很多东西。只要给你例子，你就能生成假消息，构建一个无法想象的故事。这样你足以让人工智能更加的接近于人类，真实的人工智能，这不再是一个梦想。

<h3 id="3">3. 生成对抗网络面对的挑战 </h3>

既然我们描述的生成对抗网络这么神奇且有用，那么为什么在现实的应用中并没感觉到它的广泛应用？这主要是因为我们只看到了表面的东西，同时构建一个生成对抗网络(GAN) 还有很多的障碍，并且我们还没有很好的跨越这些障碍。对于如何更好的训练[得到个好的生成对抗网络(GAN)](https://arxiv.org/pdf/1606.03498.pdf)仍然是一个非常重要的研究领域。

训练生成对抗网络最大的障碍是如何保证他的稳定性。一旦你开始训练一个生成对抗网络，如果判别器网络不够强大，那么生成器网络就无法高效的进行训练。同时还会反过来影响整个GAN的训练。从另一个角度来说，如果判别器网络对结果判定过于宽松，那么它就运行生成器生成图像时过于宽松，从而最终得到的GAN也就没有太多的用处了。

另一种观察GAN的角度，就是把GAN看成是一个整体收敛的过程。生成器网络好判别器网络相互之间对抗，并且让对方向前更近一步。同时他们也是相互依赖的，只要某一块训练失败，那么正规系统就会失败。所以我们必须保证这个歌过程不至于失败。

除此之外，还有其他的一些问题也都列在了[ppt](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf)中

* __计数问题：__  GAN无法确定在某一特定位置某一物体该出现多少次。如下图所示，GAN网络在动物的脸上放置了多个眼睛。

<div align="center">
	<img src="/assets/images/gan/count_problem.png" width="60%" height="60%"  />
</div>


* __视角问题：__ GAN不能很好的适用于3D物体。它不了解透视，分不清前视图和后视图的区别。正如下图所示，GAN将3D对象拉平成2D平面图形

<div align="center">
	<img src="/assets/images/gan/perspective_problem.png" width="60%" height="60%"  />
</div>


* __全局结构问题：__ 与视角问题类似，GANs 也无法分清物体全局结构。如下图所示，在第四幅图片中给出的一头牛，这头牛站在了它的后腿上，这在现实生活中是不可能存在的。

<div align="center">
	<img src="/assets/images/gan/structure_problem.png" width="60%" height="60%"  />
</div>


有很多研究都致力于解决这些问题。并且其中一些模型也相比之前的模型具有更好的准确性，例如：DCGAN，WassersteinGAN等等。


<h3 id="4">4. 生成对抗网络的应用</h3>

我们看到了生成对抗网络是如何运行的，并且知道它的一些缺陷。我们现在来看下生成对抗网络在实际生活中的一些应用。

* __预测视频中的下一帧：__ 你现在视频序列中训练得到一个GAN，并且让它预测下一帧视频将会发生什么事情。

<div align="center">
	<img src="/assets/images/gan/predict_vedio.png" width="50%" height="50%"  />
</div>


来自[UNSUPERVISED LEARNING OF VISUAL STRUCTURE
USING PREDICTIVE GENERATIVE NETWORKS](https://arxiv.org/pdf/1511.06380.pdf)

* __图片像素增强:__ 训练GAN，使其能将低像素图像转变成高像素图像

<div align="center">
	<img src="/assets/images/gan/high_revolution.png" width="60%" height="60%"  />
</div>

来自[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)

* __交互式图像生成：__ 画一些简单的笔画，让GAN为你生成印象深刻的图像

<div align="center">
	<img src="/assets/images/gan/interactive_image.gif" width="50%" height="50%"  />
</div>


来自： https://github.com/junyanz/iGAN

* __图像之间的转换：__ 从一张已知的图片生成另一张图片。例如：左边图像给出街道的标记，右边GAN生成真实的街景图像。右边图片例子，给定一个简单手绘手提包，然后生成一个真实手提包图片。

<div align="center">
	<img src="/assets/images/gan/image_to_image.png" width="60%" height="60%"  />
</div>

来自：[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

* __文字转换成图片:__ 只需要告诉GAN，然后让它帮你生成相应的图片

<div align="center">
	<img src="/assets/images/gan/text_to_image.png" width="60%" height="60%"  />
</div>

来自: [Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)

<h3 id="5">5. 资源</h3>

下面有一些资源，可以帮助大家更深入的了解生成对抗网络(GAN)

* [一些关于GAN的文章列表](https://github.com/zhangqianhui/AdversarialNetsPapers)
* [关于深度生成建模的介绍](http://www.deeplearningbook.org/contents/generative_models.html)
* [生成对抗网络研讨会](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks)
* [NIPS 2016关于对抗训练的研讨](https://www.youtube.com/playlist?list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF)

<h3 id="6">6. 参考文献</h3>

* [Introductory guide to Generative Adversarial Networks (GANs) and their promise!](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)

<h3 id="7">7. 附录</h3>

* [Toy GAN implementing]()