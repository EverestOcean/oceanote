---
layout: post
title:  "scikit-learn: 数据预处理1-数据/类别缺失补全"
date:   2015-11-12 09:00:00 +0800
categories: [machine-learning, scikit-learn]
---


### 缺失数据的处理

在实际应用中，缺少一个或者更多数据是普遍存在的。

许多的计算工具，无法很好的处理数据缺失的问题，从而造成一些不可预知的错误。所以在更进一步分析数据前，我们必须先处理好数据缺失的问题。


### Pandas DataFrame

为了让我们更直观的感受数据缺失问题，让我们来使用一个读取CSV文件简单的例子：

以更好的掌握问题：

```
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
df
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|NaN|8.0|
|2|0.0|11.0|12.0|NaN|

从上表可以看出，缺失的数据使用 **NaN** 进行了填充

我们可以使用 **isnull()** 来校验表格是否有值存在，若存在则值为false，若值缺失为true

```
df.isnull()
```

||A|B|C|D|
|:-:|:-:|:-:|:-:|:-:|
|0|False|False|False|False|
|1|False|False|True|False|
|2|False|False|False|True|


对于大的数据集，我们可以使用 **sum()** 方法来获得每列缺少值得个数

```
df.isnull().sum()
```

|A|0|
|:-:|:-:|
|B|0|
|C|1|
|D|1|
|dtype|int64|

我们可以使用**values** 属性来返回 **Numpy** array，从而可以输入scikit-learn模型中

```
df.values

array([1.,2.,3.,4.],
		[5.,6.,nan,8.],
		[0.,11.,12.,nan])
```

因为scikit-learn使用的是 Numpy arrary，所以使用Pandas DataFrame能很好的的将数据与scikit-learn进行对接。


### 通过pandas.DataFrame.dropnp()去除数据缺失样本/特征

对于缺失数据的样本(row)或特征(column), 我们可以通过 pandas.DataFrame.dropnp()来去除。

去除缺失数据的样本(row)

```
df 
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|NaN|8.0|
|2|0.0|11.0|12.0|NaN|

```
df.dropnp()
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|


通过设定参数 axis = 1 来去除缺失数据的特征(column)

```
df.dropnp(axis=1)
```

| | A | B |
|:-:|:-:|:-:|
|0 | 1.0 | 2.0|
|1 | 5.0| 6.0|
|2|0.0|11.0|

axis的参数可以为 {0 或者 'index'， 1 或者 'column'}

**dropnp()** 方法还有其他的参数

```
#only drop rows where all columns are NaN
df.dropnp(how='all')
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|NaN|8.0|
|2|0.0|11.0|12.0|NaN|

```
# drop rows do not have at least 4 non-NaN values
df.dropnp(thresh=4)
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|


```
# only drop rows where NaN appear in specific columns (here 'C')
df.dropnp(subset=['C'])
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|
|2|0.0|11.0|12.0|NaN|

直接删除缺失数据的样本或者特征，看起来是一个比较方便的做法，但同时也会带来一些负面的问题：

1. 有可能我们最终会删除太多样本，这将使我们的分析不可靠
2. 消除太多的特征列，存在失去一些价值的信息的风险


在下一节，我们将讨论下插值技术，这是处理缺失数据最常用的一种方法


### 通过插值技术估计缺失值

平均值插值法，通过计算特征列所有值的均值来替代缺失的数据。虽然这种方法保持了样本量并且易于使用，但是数据的变异性降低了，所以标准偏差和方差估计往往被低估。

我们使用**sklearn.preprocessing.Imputer** 方法来实现平均值插值

```
df
```

| | A| B | C | D |
|:-:|:-:|:-:|:-:|:-:|
|0|1.0|2.0|3.0|4.0|
|1|5.0|6.0|NaN|8.0|
|2|0.0|11.0|12.0|NaN|


```
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df)
imputed_data = imputer.transform(df.values)
imputed_data

array([1., 2., 3., 4.],
	   [5., 6., 7.5, 8],
		[0, 11., 12., 6.])
```

在这，我们通过计算每一列特征的平均值来替换缺失值 'NaN'

同理，我们也可以使用每一行的平均值，通过将参数 axis 设置为 1

```
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df)
imputed_data = imputer.transform(df.values)
imputed_data

array([1., 2., 3., 4.],
	   [5., 6., 6.3333333, 8],
		[0, 11., 12., 7.6666667])
```

其中对于参数 *strategy* 除了 'mean' 外还可以使用 ‘median’ 或者 ‘most_frequent’


### scikit-learn estimator API

上一节使用到的 **Imputer** 类属于scikit-learn中用于做数据转换的转换器类。

它包含两种方法：

**fit**: 该方法从训练数据中学习得到相关的参数

**transform**: 该方法使用学习到的参数来转换相应的数据

下图展示了数据转换器如何通过训练获取参数，通过如何将这些参数应用于训练数据和测试数据中

<div align="center">
<img src="/assets/images/scikit_learn/missing_data/fit-transform-scikit-learn-estimator.png" width="60%" height="60%"  />
</div>


在有监督学习中，在模型训练中除了特征外，还会提供样本对应的标签信息，在训练结束后可以使用 **predict()** 来对新数据进行预测

<div align="center">
<img src="/assets/images/scikit_learn/missing_data/fit-transform-scikit-learn-estimator.png" width="60%" height="60%"  />
</div>


图片来自:  Python machine learning by Sebastian Raschka

### 处理具有类别的数据

 
并不是所有的数据值都是数字，也有可能存在类别数据类型，例如：

1. 人类的血型：A，B，AB 和 O
2. 美国公民居住的地隶属的州名
3. T-shirt的大小 XL > L > M
4. T-shirt的颜色

即使在分裂数据中，我们可能也想进一步区分命名分类数据(nominal)和可排序的分类数据(ordinal)，例如：T-shirt的大小可以是一个序列特征，因为我们可以定义一个序列：XL > L > M


让我们来创建一个类别数据

```
import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'], 
					   ['red', 'L', 13.5, 'class2'],
					   ['blue', 'XL', 15.3, 'class3']])
df
```

| | 0 | 1 | 2| 3|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|class1|
|1|red|L|13.5|class2|
|2|blue|XL|15.3|class1|

```
df.columns
```

RangeIndex(start=0, stop=4, step=1)

```
df.colums = ['color', 'size', 'price', 'classlabel']
df
```

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|class1|
|1|red|L|13.5|class2|
|2|blue|XL|15.3|class1|

从输出结果我们可以看出，该DataFrame包含了nominal特征（color）， ordinal特征(size), numerical特征(price)。最后一列是标签数据

### ordinal 特征的处理

为了让我们的学习算法能更好的理解 ordinal 特征，我们需要将该特征转换成整型数值

但是，并没有很好的现成的函数能够自动将size特征转换成数值，所以我们需要手动定义一种映射关系。

假设不同的size特征之间存在一种关系：XL = L + 1 = M + 2.


| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|class1|
|1|red|L|13.5|class2|
|2|blue|XL|15.3|class1|

```
size_mapping = {'XL': 3, 'L': 2}
df['size'] = df['size'].mapping(size_mapping)
```

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|1|10.1|class1|
|1|red|2|13.5|class2|
|2|blue|3|15.3|class1|


如果我们想将整型数据转换成字符串，那么我们可以简单的定义一个反向映射关系字典 "inv_size_mapping" 

```
inv_size_mapping = {v:k for k, v in size_mapping.items()}
inv_size_mapping
```

{1: 'M', 2: 'L', 3: 'XL'}

```
df['size'] = df['size'].map(inv_size_mapping)
df
```

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|class1|
|1|red|L|13.5|class2|
|2|blue|XL|15.3|class1|


### 类别编码

通常，对于类别标签，我们也希望使用数值来表示

虽然大多数 scikit-learn 分类器会将类别标签转换成整数，但是最好的方法是在训练前先将相关的类别标签转换成整数，从而避免出现不必要的问题。

我们可以采用上节编码ordinal特征一样的方法，来对类别进行编码

因为类别标签不是有序的类型，所以将什么整数赋予什么样的类型标签是没有特殊的要求的。

所以可以从 0 开始对类别标签进行映射

```
np.unique(df['classlabel'])
```

array(['class1', 'class2'], dtype=object)


```
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping
```

{'class1': 0, 'class1': 1}

接下来我们使用映射关系字典，来将类别标签映射为数值

```
df['classlabel'] = df['classlabel'].map(class_mapping)
df
```

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|0|
|1|red|L|13.5|1|
|2|blue|XL|15.3|0|

正如上节中对 ’size‘ 的转换，我们同样也可以提供逆向映射字典，将整形类别标签转换成字符类别

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|0|
|1|red|L|13.5|1|
|2|blue|XL|15.3|0|

```
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
```

| | color | size | price | classlabel|
|:-:|:-:|:-:|:-:|:-:|
|0|green|M|10.1|class1|
|1|red|L|13.5|class2|
|2|blue|XL|15.3|class1|


### Nominal特征编码

如上描述，我们通过简单的字典映射将ordianl特征转换成整数。

因为在scikit-learn预估器中，我们可以使用 **LabelEncoder** 类来将相关字符串标签编码成整数

```
X = df[['color', 'size', 'price']].values
X

```

array([['green', 'M', 10.1], 
		 ['red', 'L', 13.5],
		 ['blue', 'XL', 15.3]], dtype=object)

```
color_label_encode = LabelEncoder()
X[: , 0] = color_label_encode.fit_transform(X[:, 0])
X
```

array([[1, 'M', 10.1], 
		 [2, 'L', 13.5],
		 [0, 'XL', 15.3]], dtype=object)
		 
同样我们可以采用 **one-hot encode** 来对nominal 特征进行编码, 为了实现这种编码形式，我们可以采用**scikit-learn.preprocessingOneHotEncoder**

```
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df[size] = df['size'].map(size_mapping)
X = df[['color', 'size', 'price']].values
X
```

array([['green', 1, 10.1], 
		 ['red', 2, 13.5],
		 ['blue', 3, 15.3]], dtype=object)
		 
```
color_label_encoder = LabelEncoder()
X[:, 0] = color_label_encoder.fit_transform(X[:, 0])
X
```

array([[1, 1, 10.1], 
		 [2, 2, 13.5],
		 [0, 3, 15.3]], dtype=object)
		 
```
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(categorical_feature = [0])
one_hot_encoder
```

OneHotEncoder(categorical_features=0, dtype=<type 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True)
 
```
one_hot_encoder.fit_transform(X).toarray()
```

array([[0., 1., 0., 1, 10.1], 
		 [0., 0., 1., 2, 13.5],
		 [1., 0., 0., 3, 15.3]], dtype=object)










