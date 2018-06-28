---
layout: post
title:  "迁移学习&在深度学习中使用预训练模型的艺术(译)"
date:   2017-06-01 09:00:00 +0800
categories: [deeplearning, machine-learning]
---

## 简介

神经网络是不同于其他监督学习的学习算法。为什么这么说呢？这里面有很多原因，但其中最突出的是硬件上运行算法的成本。

在现今社会中，一台机器上的RAM可以很大，也可以很便宜。你需要使用成百上千GB的RAM去运行超级复杂的监督学习算法模型--这也许是个人的一些投资。从另一个角度来说，使用GPU来计算也不是一件便宜的事情，这也可能耗费你一笔不小的资金。

当然，随着硬件的发展，在未来这种现象有可能会被改变。但是在目前的情况下，对于更好的利用已有的资源来解决深度学习的问题。特别是当我们想使用深度学习来解决生活中复杂的问题，例如: 图像和语言的识别。一旦在模型中有几个隐藏层，若想额外增加另外一层隐藏层那我们就需要耗费更多的资源。

值得庆幸的是，存在一种称为**"迁移学习"**的算法，可以让我们使用其他人已经训练好的模型，在他们的基础上做一些简单改变，然后应用到自己需要解决的问题上。因此，本文主要介绍如何在已经学习好的模型基础上来加快我们解决问题的效率。


## 目录

* [1. 什么是迁移学习](#1)
* [2. 什么是pre-trained模型](#2)
* [3. 为什么我们想使用pre-trained模型--一个实际例子](#3)
* [4. 如何使用pre-trained模型](#4)
	* [4.1 特征提取](#4.1)
	* [4.2 微调(Fine tune)模型](#4.2)
* [5. 微调(Fine tune)模型的方式](#5)
* [6. 使用pre-trained模型做字符识别](#6)
	* [6.1 只重新训练输出层](#6.1)
	* [6.2 冻结前几层的权重](#6.2)
* [7. 参考文献](#7)
* [8. 附录](#8)


<h3 id="1">1. 什么是迁移学习</h3>

让我们从老师和学生的例子中直观的感受下什么是迁移学习。

对于学生从老师的教学中获取知识来看，就是基于老师常年教育积累通过授课的手段转移给学生。

<div align="center">
<img src="/assets/images/transfer_learning/transfer-learning.jpeg" width="60%" height="60%"  />
</div>


根据这个类比，我们将其与神经网络进行对比。将神经网络用来训练某一数据，并且从中获取到相应的信息，也就是我们理解的网络权重。那么这些权重信息也可以被应用到其他的神经网络中。在这个过程中，我们不是从头开始训练其他神经网络，而是在 “迁移” 学习的基础上构建一个全新的网络。

接下来，让我们从人类进化的角度来看下迁移学习的重要性。在这采用Tim Urban在 waitbutwhy.com 上发表的理论来进一步说明下。

Tim 解释说在语言发明之前，每一代人都需要自己重新创造知识，下图展示了知识的增长是如何从一代发展到下一代的：

<div align="center">
<img src="/assets/images/transfer_learning/Knowledge-growth-graph-1b.png" width="60%" height="60%"  />
</div>


然后我们发明了语言，一种能够让知识一代一代传递下去的方式，这就是同一时间知识增长发生的变化

<div align="center">
<img src="/assets/images/transfer_learning/Knowledge-growth-graph-1b.png" width="60%" height="60%"  />
</div>


你可以看到迁移学习是多么的有用。所以，在迁移学习中传递的权重就相当与人类进化中传播知识的语言。

<h3 id="2">2. 什么是pre-trained模型</h3>

简而言之，pre-trained 模式是其他人为解决类似的问题预先构建好的模型。相比直接从零构建模型来解决相似的问题，换做使用其他人在类似问题上已经构建好的模型。

假设你想构建一个自学习的自动驾驶。你可以选择花上几年的时间来构建一个比较好的图像识别模型，反之你也可以使用Google基于ImageNet上构建好的初始化模型(一种pre-trained模型)来识别图片中的物体。

虽然使用 pre-trained 模型在你的应用中可能达不到100%的效果，但是与重新制造轮子，使用pre-trained模型可以为你提高不少的效率。接下来我们使用一些例子来为你展示如何使用迁移学习。

<h3 id="3">3. 为什么我们想使用pre-trained模型---一个实际例子</h3>

接下来是一个从手机上识别相应的主题。这是一个分类问题，其中包含4591张图片的训练数据集，和包含1200张图片的测试数据集。该问题主要的目标是将图片自动分成16个不同的类别。在做了一些图像预处理后，作者使用了如下结构的MLP(多层感知机)

<div align="center">
<img src="/assets/images/transfer_learning/mlp.png" width="60%" height="60%"  />
</div>


在将输入图像$$[224 \times 224 \times 3]$$ 拉平成 $$[150528]$$之后，作者采用过了三个包含500，500， 500个神经元的隐藏层，同时输出层使用了16个与需要预测类别相同的神经元。

最终获得了非常差的 6.8%的训练准确率。即使是对隐藏层的神经元做一些改变，也无法提高准确率。通过增加隐藏层数或者增加神经元数，也增加了每一次迭代对GPU 内存的使用时间。

下面是通过上述MLP模型输出的结果：

Epoch 10/10

50/50 [==============================] – 21s – loss: 15.0100 – acc: 0.0688

我们可以看出，最终的训练结果并没有因为增加了训练时长而得到更好的提升。因此作者尝试的使用如下的卷积神经网络(CNN)看是否能增加准确率。

<div align="center">
<img src="/assets/images/transfer_learning/cnn.png" width="60%" height="60%"  />
</div>


作者采用了3个卷积块，每个卷积块的设置如下：

1. 32个$$5*5$$ 卷积核
2. Relu激活函数
3. $$4*4$$ 最大池化层

最后将最后个卷积块的到的输出图像拉平成 [$$256$$] 的数组，并输入到一个包含60个神经元的隐藏层。接着讲隐藏层的输出输入包含16个神经元的输出层中。

最终CNN得到的结果如下：

Epoch 10/10

50/50 [==============================] – 21s – loss: 13.5733 – acc: 0.1575

虽然相比MLP在结果上有了提升，但是每一次的训练都额外增加了21秒的时间。

但是最重要的一点是，其中数据集中有个类别的数据达到了17.6%的比率，也就是如果我们把结果都预测成该类别，我们也能够得到比MLP和CNN更好的预测结果。与其采用更多的卷积块，作者开始考虑使用pre-trained模型，然后在整个网络结构中增加少数的几层。

所以作者采用了使用Keras在ImageNet上训练得到的VGG16模型，下图是作者使用到的VGG16模型的结构图：

<div align="center">
<img src="/assets/images/transfer_learning/vgg.png" width="60%" height="60%"  />
</div>


其中唯一有变化的地方是在VGG16的输出结果上增加了一个16个神经元的输出层来适应我们的问题。

最终在这个模型上得到了70%的准确率，相比直接使用MLP和CNN结果更好。除此之外，使用VGG16预训练好的模型我们大大降低了我们的训练时间。

所以，我们采用预训练好的模型，我们来进一步fine-tune VGG16模型。

<h3 id="4">4. 如何使用pre-trained模型</h3>

我们使用神经网络的主要目的是什么？我们希望通过多次的前向和后向迭代来得到准确的权重参数。我们可以通过使用之前在大数据集上pre-trained模型得到的权重参数和网络结构应用到我们的问题上。这就是我们说的迁移学习。我们可以将之前pre-trained模型应用到我们的问题上。

当然在选择pre-trained模型时，我们需要非常的注意。如果选择的pre-trained 模型与我们需要解决的问题相去甚远，那我们可能得到非常差的结果。例如：我们将使用在语音识别的模型应用到物体识别上，那也许并不是一个非常明智的选择。

值得庆幸的是，在Keras上有许多已经预训练好的模型。 ImageNet数据集被用来构建被广泛使用的模型，因为这个数据集有足够的大。其中包含了能够区分1000类别物体的模型，这1000类图片类别都是从我们日常的生活中搜集起来的，例如：猫，狗，多样的房屋，不同的交通工具。

这些已经训练好的模型可以很好的迁移到ImageNet 之外其他的数据集上做物体识别。我们通过fine-tune之前已经训练好的模型。我们假设之前训练好的模型已经被训练的非常好了，所以我们不想过快过多的修改相应的权重参数。在修改时，我们通常使用比初始学习率更低的学习率来训练。

<h3 id="5">5. 微调(Fine tune)模型的方式</h3>

1. **特征提取**， 我们可以使用pre-trained模型作为特征提取机制。我们可以做的就是去除原模型中的输出层，然后将整个网络应用于从新数据集中提取特征。
2. **使用pre-trained模型的模型结构**，我们可以使用相同的模型结构，同时随机初始化权重，并在我们的数据集上进行训练。
3. **使用整个网络中的某些层**，另一种使用预训练好的模型方法就是使用其中的一部分网络层。我们可以使用初始化的几层，然后重新训练隐藏层。我们可以通过不断的训练和测试来检验使用那些网络层。

下图也许能帮助你决定该如何更好的使用pre-trained模型：

<div align="center">
<img src="/assets/images/transfer_learning/finetune1.jpg" width="60%" height="60%"  />
</div>


**场景1 -- 现有数据集太小但是与pre-trained模型的数据有很高的相似度**，因为数据集上有很大的相似度，所以我们不需要重新训练该模型。我们需要做的就是修改输出层，从而能更好的适应现在的问题。我们使用预训练好的模型来做特征提取。假设我们希望使用在ImageNet上训练好的模型来识别新数据集中的猫或者狗时。在这数据集上有高度的相似性，但是我们只是用来识别猫和狗。所以我们修改输出层由原来的1000类修改为2类。

**场景2 -- 数据集比较小同时数据之间的相似性比较低**，在这种场景下我们可以保留初始的几层(假设是K层)，然后重新训练之后的几层。最初的几层将会被应用于新数据集上。因为数据集之间的相似性比较低，所以我们重新训练最后几层来适应我们的问题。数据过小可以用下面的方式来弥补，保持初始层的权重参数(之前在大数据集上训练的结果)。

**场景3 -- 数据集比较大但是数据相似性比较低**，因为我们拥有比较大的数据集，我们的神经网络训练将会更有效。但是，现有的数据集与预训练好模型的数据相差比较大，如果直接使用之前训练得到的模型来预测，其结果并不一定会好。因此最好的方法是在新的数据集上重新训练。

**场景4 -- 数据集比价大同时数据相似性比较高**，这是最理性的场景。在这种情况下，pre-trained 模型能得到更高的效果。在这种场景下，最好的方式保留原有模型的结构和权重，然后在新的数据集上进一步微调(fine-tune)训练。

<h3 id="6">6. 使用pre-trained模型来做字符识别</h3>

让我尝试使用pre-trained模型来解决一个简单的问题。在Keras中包含了许多在已经在ImageNet上训练好的模型。大家可以尝试下不同模型的效果。这边使用VGG16已经训练好的模型来做相关的数字字符识别。让我们看下具体符合哪个场景。我们有大约60000个手写数字训练图片。可以看出该数据集相对比较小，所以比较符合场景1和场景2。我们接下来尝试使用这两种场景下的手段来解决我们的问题。大家可以从[这里](http://yann.lecun.com/exdb/mnist/)下载到相应的字符图片数据。

1. 重新训练输出层，在这我们使用VGG16模型作为特征提取器。从我们的训练集中获取这些特征并输入到最后的输出层进行训练。同时将原来1000类输出改为10类输出。

```
# importing required libraries

from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
train=pd.read_csv("R/Data/Train/train.csv")
test=pd.read_csv("R/Data/test.csv")
train_path="R/Data/Train/Images/train/"
test_path="R/Data/Train/Images/test/"

from scipy.misc import imresize
# preparing the train dataset

train_img=[]
for i in range(len(train)):

    temp_img=image.load_img(train_path+train['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    train_img.append(temp_img)

#converting train images to array and applying mean subtraction processing

train_img=np.array(train_img) 
train_img=preprocess_input(train_img)
# applying the same procedure with the test dataset

test_img=[]
for i in range(len(test)):

    temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    test_img.append(temp_img)

test_img=np.array(test_img) 
test_img=preprocess_input(test_img)

# loading VGG16 model weights
model = VGG16(weights='imagenet', include_top=False)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_train=model.predict(train_img)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_test=model.predict(test_img)

# flattening the layers to conform to MLP input

train_x=features_train.reshape(49000,25088)
# converting target variable to array

train_y=np.asarray(train['label'])
# performing one-hot encoding for the target variable

train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)
# creating training and validation set

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.3, random_state=42)

 

# creating a mlp model
from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fitting the model 

model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))
```

2. **冻结前几层的权重**，在这我们冻结VGG16前8层的权重，然后重新训练接下来的层。主要是因为前面几层提取的是全局特征例如曲线和边缘这些在我们处理的问题中也会使用到的一些特征。希望通过保留这部分权重参数， 主要集中训练接下来几层比较精细的特征。

```
from keras.models import Sequential
from scipy.misc import imread
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

train=pd.read_csv("R/Data/Train/train.csv")
test=pd.read_csv("R/Data/test.csv")
train_path="R/Data/Train/Images/train/"
test_path="R/Data/Train/Images/test/"

from scipy.misc import imresize

train_img=[]
for i in range(len(train)):

    temp_img=image.load_img(train_path+train['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    train_img.append(temp_img)

train_img=np.array(train_img) 
train_img=preprocess_input(train_img)

test_img=[]
for i in range(len(test)):

temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))

    temp_img=image.img_to_array(temp_img)

    test_img.append(temp_img)

test_img=np.array(test_img) 
test_img=preprocess_input(test_img)


from keras.models import Model

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    model = VGG16(weights='imagenet', include_top=True)

    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    model.layers[-1].outbound_nodes = []

          x=Dense(num_classes, activation='softmax')(model.output)

    model=Model(model.input,x)

#To set the first 8 layers to non-trainable (weights will not be updated)

          for layer in model.layers[:8]:

       layer.trainable = False

# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

train_y=np.asarray(train['label'])

le = LabelEncoder()

train_y = le.fit_transform(train_y)

train_y=to_categorical(train_y)

train_y=np.array(train_y)

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.2, random_state=42)

# Example to fine-tune on 3000 samples from Cifar10

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 10 
batch_size = 16 
nb_epoch = 10

# Load our model
model = vgg16_model(img_rows, img_cols, channel, num_classes)

model.summary()
# Start Fine-tuning
model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)
```

<h3 id="7">7. 参考文献</h3>

* [1. Transfer learning & The art of using Pre-trained Models in Deep Learning](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)

