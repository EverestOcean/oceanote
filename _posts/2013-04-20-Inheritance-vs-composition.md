---
layout: post
title:  "面向对象编程中继承和组合比较"
date:   2013-04-20 09:00:00 +0800
categories: [c-plus-plus]
---

面向对象的系统中重用功能最常见的方式是类继承和对象组合。

### 类继承(Inheritance)

```
class Animal {};
class Cat : public Animal {};
```

在这个简单的例子中，类 **Cat** 是通过继承的方式与类 **Animal** 建立起了联系，因为类 **Cat** 是由 **Animal** 派生出来的。

### 对象组合(Composition)

在下面的例子中，类 **Cat** 是通过组合的方式与类 **Animal** 建立联系的，因为在类 **Cat** 存在一个指向类 **Animal** 对象的指针。在这个例子中，我们有时会称 **Cat** 为外部类(front-end class)，**Animal** 称为内部类(back-end class)。在组合关系中，外部类存放着内部类对象的一个指针。

```
class Animal {};

class Cat 
{
private:
	Animal *animal;
};
```

**类继承** 让我们用另一个类来定义一个类。

**对象组合** 跟类继承不同，新功能是通过组合对象获得的。在这种情况下，组合对象的内部细节对外部是不可见的，与此相反，继承的方式使得父类的内部细节经常是可见。


### 类继承的不足之处

虽然类继承可以更容易修改相关实现，但是子类的实现与父类的实现连接比较紧密。从而往往会造成父类的修改会使得子类也需要跟着一起修改。

让我们看下下面的例子

```
#include <iostream>
using namespace std;

class Animal 
{
public:
	int makeSound() {
		cout << "Animal is making sound" << endl;
		return 1;
	}
	
private:
	int ntimes;
}

class Cat : public Animal
{};

int main()
{
	Cat *cat = new Cat();
	cat->makeSound();
	delete cat;
	return 0;
}


```

因为 **Cat** 继承了类 **Animal** ， 所以输出的结果为：

```
Animal is making sound

```

若我们想把父类中的 **makeSound()** 修改成如下的方式：

```
Sound *makeSound(int n)
{
	cout << "Animal is making sound" << endl;
	return new Sound;
}
```
那么在 **main()** 同样需要作出改变，虽然我们用的是类**Cat** 而不是类 **Animal**

下面是对应的新代码

```
#include <iostream>

using namespace std;

class Sound {};

class Animal 
{
public:
	Sound *makeSound(int n)
	{
		cout << "Animal is making sound" << endl;
		return new Sound;
	}
private:
	int ntimes;
}

class Cat : public Animal {};

int main()
{
	Cat *cat = new Cat();
	int i = 0;
	cat->makeSound(i);
	delete cat;
	return 0;
}
```

### 对于组合方式

组合方式采用了另外一种方式让类**Cat** 重用类 **Animal** 中关于 **makeSound** 的实现。与继承 **Animal** 不同的是，类**Cat**内部通过使用 **Animal** 的对象来进行关联，同时实现自己的 **makeSound** 来调用类**Animal** 中的 **makeSound**方法。 这边是相关代码：

```
#include <iostream>
using namespace std; 

class Animal 
{
public: 
	int makeSound() {
		cout << "Animal is making sound" << endl;
		return 1;
	}
};

class Cat
{
private:
	Animal *animal;
public:
	int makeSound() {
		return animal->makeSound();
	}
};

int main() 
{
	Cat *cat = new Cat();
	cat->makeSound();
	delete cat;
	return 0;
}
```

通过使用组合的方式，子类变成了外部类，超类变成了内部类。在类继承方式中，子类自动继承了父类中非私有的方法。在组合方式中，外部类需要在内部显示的定义相应的方法来调用内部类对应的方法。此显示的调用有时称为转发或者将方法调用委托给内部类对象。

类组合的方式对代码的重用比类继承提供了更强大的封装，因为对内部类方法的修改不需要改变其他依赖外部类方法地方的代码。换句话说，继承将父类的实现细节暴露在外，通常会说 “继承打破了封装性”

对于修改类**Animal** 的方法 **makeSound()** 并不改变类**Cat** 对外提供的相应接口，所以不需要显示的修改 **main()** 函数对于的代码。

```
#include <iostream>
using namespace std; 

class Sound{};

class Animal 
{
public: 
	Sound* makeSound() {
		cout << "Animal is making sound" << endl;
		return new Sound();
	}
};

class Cat 
{
private:
	Animal *animal;
public:
	int makeSound() {
		animal->makeSound();
		return 1;
	}
};

int main() 
{
	Cat *cat = new Cat();
	cat->makeSound();
	delete cat;
	return 0;
}
``` 

由此例可以看出，内部类的修改在外部类就已经得到了很好的屏蔽。尽管内部类 **Animal** 的 **makeSound()** 方法做了修改，但是我们不需要修改 **main()** 函数中对应的代码。

对象组合模式很好的保持了类的封装性，并保证不同类专注于一类任务。对应类和类的继承想要进一步拓展就比较困难，因为这将带来很多维护的问题。

然而，基于对象组合的设计将会有更多的对象（如果更少的类），并且系统的行为将取决于它们（远程）的相互关系，而不是在一个类中定义。

相比类继承，类组合可能会更受欢迎。

### 委托(Delegation)

委托(delegation) 能让组合模式跟类继承一样强大。通过委托，两个对象同时处理一个请求。接收对象将操作委托给被委托对象，这与子类将请求提交给父类是一样的。

如下代码所示，与将类**Windows**作为类**Rectangle** 的子类不同，类**Windows**中维护一个类**Rectangle** 的对象 **rectangle**并且将相关操作委托给对象。

```
#include <iostream>
using namespace std; 

class Rectangle
{
private:
	double height, width;
public:
	Rectangle(double h, double w) {
		height = h;
		width = w;
	}
	double area() {
		cout << "Area of Rect. Window = ";
		return height*width;
	}
};

class Window 
{
public: 
	Window(Rectangle *r) : rectangle(r){}
	double area() {
		return rectangle->area();
	}
private:
	Rectangle *rectangle;
};


int main() 
{
	Window *wRect = new Window(new Rectangle(10,20));
	cout << wRect->area();

	return 0;
}
```

输出结果是：

```
Area of Rect. Window = 200
```

委托的主要优点是它可以在运行时轻松的组合行为。

例如：**Windows** 可以变成在运行时变成圆。

```
#include <iostream>
using namespace std; 

class Shape
{
public:
	virtual double area() = 0;
};

class Rectangle : public Shape
{
private:
		double height, width;
public:
	Rectangle(double h, double w) {
		height = h;
		width = w;
	}
	double area() {
		return height*width;
	}
};

class Circle : public Shape
{
private:
		double radius;
public:
	Circle(double r) {
		radius = r;
	}
	double area() {
		return 3.14*radius*radius;
	}
};

class Window 
{
public: 
	Window (Shape *s):shape(s){}
	double area() {
		return shape->area();
	}
private:
	Shape *shape;
};


int main() 
{
	Window *wRect = new Window(new Rectangle(10,20));
	Window *wCirc = new Window(new Circle(20));
	cout << "rectangular Window:" << wRect->area() << endl;
	cout << "circular Window:" << wCirc->area() << endl;
	return 0;
}
```