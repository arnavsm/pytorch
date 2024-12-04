# Object-Oriented Programming in Python

## Introduction to OOP

Object-Oriented Programming (OOP) is a programming paradigm that uses objects and classes to organize and structure code. In Python, everything is an object, making it an ideal language for OOP.

## Key Concepts

### 1. Classes and Objects

#### What is a Class?
- A class is a blueprint for creating objects
- Defines attributes (data) and methods (functions) that the objects will have
- Acts as a template for creating instances

#### What is an Object?
- An instance of a class
- A concrete entity created from a class blueprint
- Contains its own set of attributes and can perform methods defined in the class

#### Basic Class Definition
```python
class MyClass:
    def __init__(self, parameter1, parameter2):
        # Constructor method
        self.attribute1 = parameter1
        self.attribute2 = parameter2
    
    def my_method(self):
        # Class method
        return f"Doing something with {self.attribute1}"
```

### 2. Constructors

#### The `__init__` Method
- Special method called when an object is created
- Used to initialize object attributes
- `self` parameter refers to the instance being created

```python
class Person:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age
```

### 3. Access Modifiers

#### Public, Private, and Protected Attributes
- Python uses naming conventions to indicate attribute accessibility
  - Public: Standard attributes (no prefix)
  - Protected: Single underscore prefix `_attribute`
  - Private: Double underscore prefix `__attribute`

```python
class Example:
    def __init__(self):
        self.public_attr = 1        # Public
        self._protected_attr = 2    # Protected (convention)
        self.__private_attr = 3     # Private (name mangling)
```

### 4. Inheritance

#### Types of Inheritance
1. **Single Inheritance**: One parent class
2. **Multiple Inheritance**: Multiple parent classes
3. **Multilevel Inheritance**: Inherited through multiple levels
4. **Hierarchical Inheritance**: Multiple child classes from one parent
5. **Hybrid Inheritance**: Combination of inheritance types

#### Basic Inheritance
```python
class Parent:
    def parent_method(self):
        pass

class Child(Parent):
    def child_method(self):
        pass
```

#### Using `super()`
- Allows calling methods from the parent class
- Useful in method overriding and constructor chaining

```python
class Child(Parent):
    def __init__(self, new_param):
        super().__init__()  # Call parent class constructor
        self.new_param = new_param
```

### 5. Polymorphism

#### Method Overriding
- Redefining a method in the child class that exists in the parent class
- The child class method takes precedence

```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):  # Overrides parent method
        print("Dog barks")
```

### 6. Special/Magic Methods (Dunder Methods)

#### Common Magic Methods
- `__init__()`: Constructor
- `__str__()`: String representation
- `__repr__()`: Official string representation
- `__len__()`: Length of object
- `__add__()`: Addition operation
- `__eq__()`: Equality comparison

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
```

### 7. Class and Static Methods

#### Class Methods
- Uses `@classmethod` decorator
- Takes class as first parameter instead of instance
- Can modify class state
- Useful for alternative constructors

```python
class MyClass:
    class_attribute = 0
    
    @classmethod
    def create_from_something(cls, param):
        return cls(special_processing(param))
```

#### Static Methods
- Uses `@staticmethod` decorator
- Doesn't receive class or instance as first parameter
- Used for utility functions related to the class

```python
class MathOperations:
    @staticmethod
    def add(x, y):
        return x + y
```

### 8. Composition vs Inheritance

#### Composition (Has-A Relationship)
- Creating complex objects by combining simpler objects
- More flexible than inheritance

```python
class Engine:
    def start(self):
        print("Engine started")

class Car:
    def __init__(self):
        self.engine = Engine()  # Composition
    
    def start_car(self):
        self.engine.start()
```

### 9. Abstract Base Classes

#### Using `abc` Module
- Define interfaces and ensure method implementation
- Cannot be instantiated directly

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def area(self):
        # Must implement abstract method
        return self.width * self.height
```

## Best Practices

1. Follow naming conventions
   - CamelCase for class names
   - snake_case for methods and attributes
2. Keep classes focused and modular
3. Use composition over inheritance when possible
4. Avoid deep inheritance hierarchies
5. Use type hints for better code readability
6. Document your classes and methods

## Common Pitfalls

- Overusing inheritance
- Creating overly complex class hierarchies
- Not understanding the difference between class and instance attributes
- Misusing magic methods
- Ignoring Python's "we're all consenting adults" philosophy of access control

## Conclusion

Python's object-oriented programming provides powerful and flexible ways to structure code. Understanding these concepts allows you to write more organized, reusable, and maintainable software.