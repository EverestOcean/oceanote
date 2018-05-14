---
layout: post
title:  "理解Python元类(Metaclass)"
date:   2018-05-13 13:00:00 +0800
categories: [python]
---
## 目录
* [1. 理解类也是对象](#1)

* [2. 动态的创建类](#2)

* [3. 元类(Metaclass)](#3)

	* [3.1 什么是元类](#3.1) 

	* [3.2 使用\_\_metaclass\_\_ 属性](#3.2)

	* [3.3 自定义元类](#3.3)

* [4. 为什么用元类替代函数](#4)

* [5. 为什么要使用元类](#5)
	
<h2 id="1">1. 理解类也是对象</h2>

在理解Python元类之前，你首先需要掌握下Python中的类。 Python中类的概念借鉴与Smalltalk。在大多数语言中，类是一组用来描述如何生成一个对象的代码段。 这一点在Python中任然成立

```
>>> class ObjectCreator(object):
...     pass
... 
>>> my_obj = ObjectCreator()
>>> print(my_obj)
<__main__.ObjectCretor object at 0x10981d990>
>>>

```
但是在Python中类远不止如此。类同样也是一种对象。只要你使用关键字class，Python解释器在执行的时候就会创建一个对象。下面的代码段：

```
>>> class ObjectCreator(object):
...     pass
...

```
将在内存中创建一个对象，名字就是ObjectCreator。这个对象（类）自身拥有创建对象（类实例）的能力，而这就是为什么他是一个类的原因。但本质上它还是一个对象，从而你可以对它做如下的操作：

```
>>> class ObjectCreator(object):
...     pass
...

```

* 你可以将它赋给一个变量
* 你可以拷贝它
* 你可以给它增加相关属性
* 你可以将它作为一个函数的变量

如下所示：

```
>>> print(ObjectCreator) # you can print the class because it also a object
<class '__main__.ObjectCreator'>
>>> def echo(o):
...	    print(o)
... 
>>> echo(ObjectCreator) # you can pass a class as a parameter
<class '__main__.ObjectCreator'>
>>> print(hasattr(ObjectCreator, 'new_attribute'))
False
>>> ObjectCreator.new_attribute = 'foo' # you can add attribute to a class
>>> print(hasattr(ObjectCreator, 'new_attribute'))
True
>>> print(ObjectCreator.new_attribute)
foo
>>> ObjectCreatorMirror = ObjectCreator
>>> print(ObjectCreatorMirror.new_attribute)
foo
>>> print(ObjectCreatorMirror())
<__main__.ObjectCretor object at 0x10981da10>
```

<h2 id="2"> 2. 动态的创建类</h2>

因为类也是对象，所以你可以在运行时动态的创建它们，就像其他任何对象一样。首先你可以在函数中创建类，使用class关键字就可以。

```
>>> def choose_class(name):
...     if name == "foo":
...         class Foo(object):
... 	  	      pass
... 		  return Foo # return the class, not instance
... 	else: 
... 		  class Bar(object):
... 			   pass
... 		  return Bar
>>> 
>>> Myclass = choose_class('foo')
>>> print(Myclass) # the function return a class, not a instance
<class '__main__.Foo'>
>>> print(Myclass()) # you can create an object from this class
<__main__.Foo object at 0x89c6d4c>
```

但这还是不够动态，因为你需要自己编写整个类的代码。 由于类也是对象，因此它们也需要什么东西来生成才对。当你你使用class 关键字时，Python解释器会自动创建这个对象。 但就和Python中的大多数事情一样，Pthon任然提供给你手动处理的方法。还记得内建函数type吗？ 这个古老但强大的函数能够让你知道一个对象的类型是什么：

```
>>> print(type(1))
<type 'int'>
>>> print(type("1"))
<type 'str'>
>>> print(ObjectCreator)
<type 'type'>
>>> print(type(ObjectCreator()))
<class '__main__.ObjectCreator'>
```

同样，type 具有一种完全不同的能力，它也能动态的创建类。type可以将类的描述作为参数，然后生成一个类：

```
type(name of the class, tuple of the parent class (for inheritance, can be empty), dictonary containing attributes names and values)
```
例如：

```
>>> class MyShinyClass(object):
...     pass

```
能通过下面的方式进行创建：

```
>>> MyShinyClass = type('MyShinyClass', (), {})
>>> print(MyShinyClass)
<class '__main__.MyShinyClass'>
>>> print(MyShinyClass()) # Create a instance with the class
<__main__.MyShinyClass object at 0x8997cec>
```

你会发现我们使用”MyShinyClass“作为类名，并且也可以当做一个变量来作为类的引用。

接下来我们来看下如何使用type来创建类的属性：

```
>>> class Foo(object):
...     bar = True

```

可以用以下方式创建：

```
>>> print(Foo)
<class '__main__.Foo'>
>>> foo = Foo()
>>> print(foo)
<__main__.Foo object at 0x8a9b84c>
>>> print(foo.bar)
True

```
同时FooChild 可以继承Foo类：

``` 
>>> class FooChild(Foo):
...     pass
```

使用type 进行创建：

```
>>> FooChild = type('FooChild', (Foo, ), {})
>>> print(FooChild)
<class '__main__.FooChild'>
>>> print(FooChild.bar)
True
```
即时你想为FooChild类增加相应的方法，只需要定义好相应的函数，并将该函数作为属性传入：

```
>>> def echo_bar(self):
...     print(self.bar)
>>> FooChild = type('FooChild', (Foo, ), {'echo_bar': echo_bar})
>>> hasattr(Foo, 'echo_bar')
False
>>> hasattr(FooChild, 'echo_bar')
True
>>> my_foo = FooChild()
>>> my_foo.echo_bar()
True

```
并且你也可以将更多的方法，在你动态创建一个类之后。就像将方法添加到通过普通方法创建的类一样

```
>>> def echo_bar_more(self)
...    print("yet another method")
>>> FooChild.echo_bar_more = echo_bar_more
>>> hasattr(FooChild, 'echo_bar_more')
True
```

可以看到，在Python中类也可以是对象，你可以动态的创建类。这就是当我们使用关键字class时Python在幕后做的事情，而这就是通过元类来实现的。



<h2 id="3">3. 元类 </h2>


<h3 id="3.1">3.1 什么是元类 </h3>

通过上面的描述，类也是对象， 从而可以知道元类就是创建类的类, 如图1：
![图1](https://github.com/everestocean/oceanote/blob/gh-pages/static/img/_posts/python_metaclass/python_metaclass_picture1.png)

你定义一个类的目的主要是为了创建一个对象，对吗？ 那么我们知道Python class 也是对象。 那么，元类（metaclass）就是用来创建这些对象的类。你可以通过下面的表达式想象下：

```
>>> Myclass = Metaclass()
>>> m_object = Myclass()
```

我们知道 type 可以让我们通过下面的方式创建类：

```
Myclass = type('Myclass', (), {})
```

这主要是因为type实际上是一个元类(metaclass)。 type是Python中在幕后用来创建所有类的元类。

题外话，为什么 type 不用大写的Type表示呢？也许是为了保持与 ‘str’， ‘int’ 这些类的统一吧。 ‘str’是用来创建字符串的类，‘int’是用来创建整形的类。

你可以通过 \_\_class\_\_来验证，在Python中，所有的东西都是一个类，对的，是所有的东西都是对象。包括 int(整型数)，string(字符串)， 函数以及类。它们都是对象，而且他们都是从一个类创建出来的。

```
>>> age = 35
>>> age.__class__
<type 'int'>
>>> name = 'bob'
>>> age.__class__
<type 'str'>
>>> def foo():
...     pass
>>> foo.__class__
<type 'function'>
>>> class Bar(object):
...     pass
>>> b = Bar()
>>> b.__class__
<type '__main__.Bar'>

```

那么对于\_\_class\_\_ 的 \_\_class\_\_ 又是什么呢？

```
>>> age.__class__.__class__
<type 'type'>
>>> name.__class__.__class__
<type 'type'>
>>> foo.__class__.__class__
<type 'type'>
>>> b.__class__.__class__
<type 'type'>

```

因此，元类是创建类这种对象的东西，type是Python的内建元类，当然你也可以自己创建元类。


<h3 id="3.2">3.2 __metaclass__ 的属性</h3>

你可以在写一个类的时候为其添加 \_\_metaclass\_\_ 属性， 定义了\_\_metaclass\_\_就定义了这个类的元类。

```
>>> class Foo(object):
...     __metaclass__ = something...
... [...]

``` 
