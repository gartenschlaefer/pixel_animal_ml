# --
# task 0: start here with the preliminaries

# Note that this task does not belong to the overall assignment but helps you getting started.
# If you are already and experienced python user you can skip this task, just consider installing the required packages.

# First you have to setup your python environment on your operating system.
# If python is installed you should be able to enter its interpreter via the command line (symbol used here for command line: >)
# > python

# If you see a python version starting with 3.x.x then you are fine otherwise install any of the latest 3.x.x python versions

# Also check if pip (python package installer) is installed (you can also use conda for installation but pip is usually the better choice):
# > pip --version

# Next it is recommended to create a virtual python environment so that only necessary packages for the specific tasks are installed.
# This can be done via:
# > python -m venv "path/to/your/environment"

# To activate this environment (has to be done each time you open a new terminal) type
# on Linux:
# > source "path/to/your/environment/bin/activate"

# deactivate an active environment with
# > deactivate

# Usually a python project contains a file with a list of required packages.
# This file used for pip installs is named "requirements.txt".
# Make sure that your virtual environment is enabled (usually you see it on the command line on the left), then the packages can be installed via:
# > pip install -r requirements.txt

# Now hopefully all packages necessary for this assignment are installed.
# More importantly I would like to mention that you can use any programming editor or IDE you like the most.
# However, the best is to start with a simple editor, so that you know what the IDE does in the background.
# I can suggest the sublime editor and the terminal for running the code.
# Finally, lets try it out and learn some basics of python programming...

# python runs its code from top to bottom

# define a variable
test_var = 1

# with assert a condition can be checked
assert test_var == 1, "This text is printed if the condition is not fulfilled"

# print variables
print("The value of your variable is: ", test_var)

# package import of numpy (with an alias -> as np)
import numpy as np

# numpy array
a = np.array([[1, 2], [3, 4]])

# print is a powerful function to display variables and references in the terminal, use it as much as possible
print("a", 1, 2, 3, 4, "{} end".format('the'))

# print that variable and note that a numpy array has a function to get its shape (very useful)
print("np array: \n{}\nwith shape: {}\nindexing first row: {}".format(a, a.shape, a[0, :]))


# ***
# your code here

# this is a not implemented error, you should simply remove it and do some coding here
raise NotImplementedError()

# variables which are defined as None in the coding section have to be assigned to its correct value or reference
# create another array b with random numbers (look for the function numpy.random.randn)
b = None

# element wise multiply a with b and store the result in c
c = None

# 
# ***


# check if it might be correct
assert isinstance(c, np.ndarray) and c.shape == (2, 2), "check data type or shape of c"


# ***
# your code here

# now do the same array multiplication with the torch framework
import torch

raise NotImplementedError()

# array
a = None

# create another array b with random numbers
b = None

# element wise multiply a with b and store the result in c
c = None

# 
# ***


# check if it might be correct
assert torch.is_tensor(c) and c.shape == (2, 2), "check data type or shape of c"


# often a config file is useful in a project to set parameters for your components. Here we use a .yaml file as format.
import yaml

# your config file loaded as python dictionary
cfg = yaml.safe_load(open("./config.yaml"))['task0']

# a dictionary is a very widely used data structure in python and contains of
# keys
print("keys: ", cfg.keys())
# and values
print("values: ", cfg.values())

# it can be created like this
d = {'a': 1, 'b': 2}

# and accessed like this
print(d['a'])


# ***
# your code here

raise NotImplementedError()

# access the important_variable in task0 of the config.yaml file
important_variable = None

#
# ***


#assert not important_variable is None, 'something wrong'
print("The variable is: ", important_variable)

# dictionaries can also be accessed with the unpacking operator (**) for instance to fuse multiple dictionaries
print("fusing of dictionaries", {**{1: 1, 2: 2}, **{3: 3}})


# ***
# your code here

raise NotImplementedError()

# fuse the dictionaries another_dict and yet_another_dict
fused_dictionaries = None

#
# ***


# this is the python equivalent to a main function which can be seen as the start of your program
# the only difference to a real main function is that it is only called when the program is run directly and not imported by another python file
# best practice is to define a function "def main()" but it is only necessary if you intend to import this main function somewhere else
# (still note that python scripts runs the code from top to bottom, functions are stored but not executed when they are defined)

def main():
  pass

if __name__ == '__main__':
  main()

# all other code should be usually placed in functions or classes (this demo is therefore not the best coding practice)
def foo(x, *args, **kwargs):
  print("foo function")
  print("args: ", args)
  print("kwargs: ", kwargs)
  return x + 1

# *args are an arbitrary number of arguments and stored in lists, e.g. the print function uses args
args = ['arg1', 2, [4, 2]]

# **kwargs are an arbitrary number of keyword arguments
kwargs = {'a': 1, 'b': 2}

# (also a function, but one line)
bar = lambda x: x + 2

# also note that a variable can also be a reference to a function
function_pointer_foo = foo

# call of functions
a = foo(1, *args, **kwargs)
b = bar(1)
c = function_pointer_foo(1)
print('results: {}, {}, {}'.format(a, b, c))

# classes are very convenient because they store variables in one object, also they can be inherited
class Animal():
  def __init__(self):
    self.vocal = 'abstract animal sound'

  def vocalize(self):
    print(self.vocal)

# an inherited class looks like this
class Dog(Animal):
  def __init__(self):
    # the parent (Animal) __init__ function is called
    super().__init__()
    # overwrite the member variable
    self.vocal = 'wuff wuff!!"'
  def get_vocal(self):
    return self.vocal

# a instance can be created like this
animal = Animal()
dog = Dog()

# call a member function in the class
animal.vocalize()
dog.vocalize()

# get a member variable is also possible... 
print("the dog says: ", dog.vocal)
# but its always better to define a getter function
print("the dog says: ", dog.get_vocal())


# ***
# your code here

raise NotImplementedError()

# create a cat class from Animal and display its vocalization

#
# ***


# a very useful package is pathlib and its Path class
from pathlib import Path

# create a path
p = Path('./')

# get all python files of this path recursively
pyfiles = p.glob('**/*.py')

# a cool operator for paths
new_p = p / 'experiments/'

# many more cool path members helping to access the information of a filename
print(new_p.parent)
print(new_p.stem)

# make them a sorted list and print them
print("python files: ", sorted(list(pyfiles)))

# ***
# your code here

raise NotImplementedError()

# print all yaml files of the new_p

#
# ***


# another distinct functionality in python are generated lists
gen_list = [i for i in range(10) if i % 2]
print("gen list: ", gen_list)

# if you like one-liners then this is the best way to use them (also works for dictionaries):
d = {'class{}'.format(i): entry for i, entry in enumerate(gen_list) if entry != 1}
print("gen dict: ", d)


# ***
# your code here

raise NotImplementedError()

# create a list only with even entries

#
# ***


# if it comes to accessing stored data (e.g. in datamodules), it is important to understand the iterator objects
# an iterator is a class that has following member functions:
class MyIterator():
  def __iter__(self):
    self.a = np.arange(5)
    self.i = 0
    return self

  def __next__(self):
    if self.i >= len(self.a): 
      raise StopIteration
      return self
    res = self.a[self.i]
    self.i += 1
    return res

# init an iterator
it_class = MyIterator()
it = iter(it_class)
print("iterator:")
print(next(it))
print(next(it))

# or run it in a loop
[print(i) for i in MyIterator()]


# ***
# your code here

raise NotImplementedError()

# create an iterator like the previous one but it should get the items from the back of the list to the first element

#
# ***


# a more advanced technique which enables more flexible coding, is to get functions by their string name
# we use the importlib package
import importlib

# import the module of the file './plots.py' (its the file name of a python file when its in a system path)
module = importlib.import_module('plots')
  
# get a function or class from the module, this actually is the equivalent of "from plots import plot_2dmatrix as fp_plot"
fp_plot = getattr(module, 'plot_2dmatrix')
print("plot function: ", fp_plot)

# use this function
fp_plot([np.arange(4).reshape(2, 2), np.arange(4, 8).reshape(2, 2)])

# if you cannot find the requested module check your python paths with
import sys
print(sys.path)

# and append a new path to the list, e.g. a relative path
sys.path.append('../')
print(sys.path)


# ***
# your code here

raise NotImplementedError()

# get the function to_short_class_name from plots
fp_short = None

classes = ['dog', 'frog', 'horse']
class_nr = [0, 1, 2, 0, 2, 1]

# change the class names from integer class numbers
short_class_names = [fp_short(class_nr, classes) for class_nr in class_nr]

print(short_class_names)

#
# ***


# this is the end of the short introduction to python specific coding
# please also checkout other material to learn more on this subject
# also recommend something if it is missing here
# Good luck for the assignment - gartenschlaefer 


print("Everything successful!")