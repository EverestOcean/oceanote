---
layout: post
title:  "理解Python元类(Metaclass)"
date:   2018-05-13 13:00:00 +0800
categories: [python]
---

这篇文章是从[Stack Overflow](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)上讨论Python 元类的帖子中一个非常热的回答中照搬过来的。同时借鉴了CSDN类似文章 [深刻理解Python中的元类(metaclass)以及元类实现单例模式](https://www.cnblogs.com/tkqasn/p/6524879.html)。在这需要感谢以上文章作者的分享，同时在自己博客中记录的目的：一是通过书写能让自己加深对 Python 元类的理解，二是为了保留，方便以后的回顾和再学习。

## 目录
* [1. 理解类也是对象](#1)

* [2. 动态的创建类](#2)

* [3. 元类(Metaclass)](#3)

	* [3.1 什么是元类](#3.1) 

	* [3.2 使用\_\_metaclass\_\_ 属性](#3.2)

	* [3.3 自定义元类](#3.3)
* [参考文献](#4)

	
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
例如：当我们写如下代码时：

```
>>> class Foo(Bar):
...     pass

```
在该类定义的时候，并未在内存中生成，直到它被调用， Python做了如下的操作：

1. Foo中有\_\_metaclass\_\_属性吗？如果是，Python会在内存中通过\_\_metaclass\_\_创建一个名字为Foo的类对象（这边是类对象，请注意）
2. 如果Python没有找到\_\_metaclass\_\_, 它会继续在父类中寻找\_\_metaclass\_\_属性，并尝试做和前面同样的操作
3. 如果Python在任何父类中都找不到\_\_metaclass\_\_， 它就会在模块层次中去寻找\_\_metaclass\_\_， 并尝试做同样的操作
4. 如果还是找不到\_\_metaclass\_\_， Python会用内置的type来创建这个类对象

<h3 id="3.3">3.3 自定义元类 </h3>

使用元类的主要目的是在创建类的时能自动改变类。 通常，你会为API做这样的事情，你希望可以创建符合上下文的类。

试想一个非常简单的例子，如果你想在模块里保证类的所有属性都应该是大写的形式。有好几种办法可以办到，但其中一种就是通过设定元类\_\_metaclass\_\_。采用这种方法，模块中所有的类都会通过这个类来创建，我们只需要告诉元类，所有的属性都改写成大写形式就万事大吉了。

**使用函数作为元类**

```
# the metaclass will automatically get passed the same argument
# that you usually pass to `type`
def upper_attr(future_class_name, future_class_parents, future_class_attr):
    """
      Return a class object, with the list of its attribute turned
      into uppercase.
    """

    # pick up any attribute that doesn't start with '__' and uppercase it
    uppercase_attr = {}
    for name, val in future_class_attr.items():
        if not name.startswith('__'):
            uppercase_attr[name.upper()] = val
        else:
            uppercase_attr[name] = val

    # let `type` do the class creation
    return type(future_class_name, future_class_parents, uppercase_attr)

__metaclass__ = upper_attr # this will affect all classes in the module

class Foo(): # global __metaclass__ won't work with "object" though
    # but we can define __metaclass__ here instead to affect only this class
    # and this will work with "object" children
    bar = 'bip'

print(hasattr(Foo, 'bar'))
# Out: False
print(hasattr(Foo, 'BAR'))
# Out: True

f = Foo()
print(f.BAR)
# Out: 'bip'

```

**使用自定义的类作为元类**

```
# remember that `type` is actually a class like `str` and `int`
# so you can inherit from it
class UpperAttrMetaclass(type):
    # __new__ is the method called before __init__
    # it's the method that creates the object and returns it
    # while __init__ just initializes the object passed as parameter
    # you rarely use __new__, except when you want to control how the object
    # is created.
    # here the created object is the class, and we want to customize it
    # so we override __new__
    # you can do some stuff in __init__ too if you wish
    # some advanced use involves overriding __call__ as well, but we won't
    # see this
    def __new__(upperattr_metaclass, future_class_name,
                future_class_parents, future_class_attr):

        uppercase_attr = {}
        for name, val in future_class_attr.items():
            if not name.startswith('__'):
                uppercase_attr[name.upper()] = val
            else:
                uppercase_attr[name] = val

        return type(future_class_name, future_class_parents, uppercase_attr)
```

但是，这种方式其实不是OOP。我们直接调用了type，而且我们没有改写父类的__new__方法。现在让我们这样去处理:

```
class UpperAttrMetaclass(type):

    def __new__(upperattr_metaclass, future_class_name,
                future_class_parents, future_class_attr):

        uppercase_attr = {}
        for name, val in future_class_attr.items():
            if not name.startswith('__'):
                uppercase_attr[name.upper()] = val
            else:
                uppercase_attr[name] = val

        # reuse the type.__new__ method
        # this is basic OOP, nothing magic in there
        return type.__new__(upperattr_metaclass, future_class_name,
                            future_class_parents, uppercase_attr)
```

你可能已经注意到了有个额外的参数upperattr_metaclass，这并没有什么特别的。类方法的第一个参数总是表示当前的实例，就像在普通的类方法中的self参数一样。当然了，为了清晰起见，这里的名字我起的比较长。但是就像self一样，所有的参数都有它们的传统名称。因此，在真实的产品代码中一个元类应该是像这样的：

```
class UpperAttrMetaclass(type):

    def __new__(cls, clsname, bases, dct):

        uppercase_attr = {}
        for name, val in dct.items():
            if not name.startswith('__'):
                uppercase_attr[name.upper()] = val
            else:
                uppercase_attr[name] = val

        return type.__new__(cls, clsname, bases, uppercase_attr)
```

如果使用super方法的话，我们还可以使它变得更清晰一些。

```
class UpperAttrMetaclass(type):

    def __new__(cls, clsname, bases, dct):

        uppercase_attr = {}
        for name, val in dct.items():
            if not name.startswith('__'):
                uppercase_attr[name.upper()] = val
            else:
                uppercase_attr[name] = val

        return super(UpperAttrMetaclass, cls).__new__(cls, clsname, bases, uppercase_attr)
```

<h2 id="4"> 参考文献 </h2>
[1. What are metaclasses in Python?](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)

[2. 深刻理解Python中的元类(metaclass)以及元类实现单例模式](https://www.cnblogs.com/tkqasn/p/6524879.html)

[3. 深刻理解Python中的元类(metaclass)](http://blog.jobbole.com/21351/)