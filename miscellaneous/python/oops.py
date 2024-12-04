# check oops.cpp



# Class name should be in Pascal case (similar to camel case)
# and its attirbutes & methods in underscore from

# Special/Magic/Dunder methods start and end with double underscore


# Class 1
class ATM:

    # There are instance values which are different for each object
    # Static/class values are same or shared for each object of same class
    # Static variables are always outside constructor
    # Every other functionality is same as that of instance variables
    __counter = 1 # Can be accessed by self.sno = ATM.counter instance variable as below

    def __init__(self, pin, balance): # self is an instance of class, hence object

        self.pin = pin
        self.balance = balance

        self.sno = ATM.__counter
        ATM.__counter = ATM.__counter + 1
        # Above we have set the access modifier to public
        # If we want it to be private, we change it to self.__pin & self.__balance
        # Basically a double underscore before the actual variable
        # This makes the variable _ATM__pin
        
        # However nothing in python is truly private
        # What we did above is like a "gentlemen's agreement"
        # we can access it now by sbi._ATM__pin & similarly balance 
        self.__menu()
        # similarly we can change it to __menu

    def __menu(self):
        user_input = int(input("Hello, how would you like to proceed? \n1. Enter 1 to create pin \n2. Enter 2 to deposit \n3. Enter 3 to withdraw \n4. Enter 4 to check balance \n5. Enter 5 to exit"))

        if user_input == 1:
            self.create_pin()
        if user_input == 2:
            print("Create pin")         
        if user_input == 1:
            print("Create pin")
        if user_input == 1:
            print("Create pin")
        if user_input == 1:
            print("Create pin")

    def create_pin(self):
        self.pin = int(input("Enter pin: "))
        print("Pin Created")

    def deposit(self):
        pin_entered = int(input("Enter pin: "))
        if self.pin == pin_entered:
            amount = int(input("Enter amount: "))
            self.balance = self.balance + amount
            print("Deposited successfully")


    # Different Methods below
            

    def example(atm): # pass by reference (basically an object as parameter)
        if atm.balance == atm._ATM__pin:
            pass

        # obviously only to mutable data types

    
    def get_counter(): # as it is a static variable, no self (no instance, no object)
        return ATM.__counter

# Class 2
            
class fraction:

    def __init__(self, num, denom):
        self.numerator = num
        self.denominator = denom

    def __str__(self):
        return (f"{self.numerator}/{self.denominator}") 
    
    # Incase we want to add 2 fractions

    def __add__(self, other):

        temp_num = self.numerator * other.denominator + self.denominator * other.numerator
        temp_denom = other.denominator * self.denominator

        return (f"{temp_num}/{temp_denom}")

# As we know, list is a class like everything in python
# L = [1,2,3] or you can do L = list()"""

frac1 = fraction(5, 6)
print(frac1)

# In Python, every data element or object stored in the memory is been referenced by a 
# numeric value. These numeric values are useful in order to distinguish them from other values.
# This is referred to as the identity of an object. The python id() function is used to return
# a unique identification value of the object stored in the memory. This is quite similar to how 
# unique memory addresses are assigned to each variable and object in the C programming language.

print(id(frac1)) 

# Two types of relationships
# Aggregation & Inheritance
# (has)         (is)

# Aggregation
class Customer:
    def __init__(self, n, ag, add):
        self.name = n
        self.age = ag
        self.address = add
    
    def edit_profile(self, new_name, new_age, new_city, new_pin, new_state):
        self.name = new_name
        # etc.
        self.address.change_address(new_city, new_pin, new_state)

class Address:
    def __init__(self, city, pincode, state):
        self.city = city
        self.pincode = pincode
        self.state = state

    def change_address(self, new_city, new_pin, new_state):
        self.city = new_city
        self.pincode = new_pin
        self.state = new_state



# Inheritance        
class user:
    def __init(self, name, age, id):
        self.name = name
        self.age = age
        self.id = id

    def login(self):
        pass

    def register(self):
        pass

class student(user):

    def enroll(self):
        pass

    def review(self):
        pass

    def register(self):
        super().register() # instead of method overriding it executes parent class method
        # super() doesnt work outside in global only here inside child class


class instructor(user):
    pass


# student (child class) that can access all the methods of user (parent class) but vice versa is
# not true. Child class cannot access hidden member (private variables) of parent class


# Polymorphism
# if methods of parent class and child class conflict then child class method is preferred or 
# rather, executed


# If constuctor of child class is made then instructor of child is done we need parent variables in child
# constructor as only one constructor is executed.
# we can do this by super().__init__(#parameters#)
# Because of this we can access parent variables, methods from child methods






# Types of Inheritance
# 1. Single-level (1 parent class 1 child class) 
# 2. Multi-level (1 parent 1 grandparent 1 child)
# 3. Heirarchical (2 childs 1 parent)
# 4. Mulitple (2 parents 1 child)
# 5. Hybrid (combination of above)

