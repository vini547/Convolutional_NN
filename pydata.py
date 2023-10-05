import numpy as np
import tensorflow as tf
import scipy as sp
import matplotlib as mpl
import time as time

START = time.time()
def first_generator():
    x = int(input("Please enter a value for the number of indexes in the First Activation Layer: "))
    
    vec = []
    for i in range(x):
        vec += [round(np.random.uniform(0.0,1.0),2)]
    first_generator = np.array(vec)    
    return first_generator

def bias_first_activation(a): 
    buff = []
    for i in range(int(a.shape[0])):
        buff += [round(np.random.uniform(0.0,10.0),2)]
    bias_first_activation = np.array(buff)
    return bias_first_activation



def transposed_weights_matrix_constructor(np_array):
    lenght = len(np_array)
    x = int(input("Please enter a value for the Number of Neurons in the Hidden Layer: "))      
    transposed_weights_matrix_constructor = np.array(np.zeros((lenght,x)))
    
    for i in range(int(transposed_weights_matrix_constructor.shape[0])):
        for j in range(int(transposed_weights_matrix_constructor.shape[1])):
            transposed_weights_matrix_constructor[i][j] = round(np.random.uniform(1.0,9.0),2)

    return transposed_weights_matrix_constructor


""" def weighted_sum(a, b):
  e =[]
  for i in a.shape[0]:
    for j in b.shape[1]:
        element_a = a[i][j]
        element_b = b[i][j]
        product = element_a * element_b
        e += product
        
  return e      """ 

def weighted_sum(a, b):
  b2 = b.transpose()
  print(b2)
  e = np.sum(a * b2)        
  return e  


def initialize_network():
    a = first_generator()
    b = transposed_weights_matrix_constructor(a)
    h = weighted_sum(a,b)
    g = bias_first_activation(h)
    w = np.add(h,g)
    sig = tf.constant(w)
    sigmoid = tf.sigmoid(sig)
    print(sigmoid)

    

np.set_printoptions(precision=6, suppress=True, linewidth=200)
a = first_generator()
b = transposed_weights_matrix_constructor(a)
h = weighted_sum(a,b)
print("first_generator", end='\n')
print(a, end='\n')
print(type(a), end='\n')
print('\n')
print("transposed_weights_matrix_constructor", end='\n')
print(b, end='\n')
print(type(b), end='\n')
print('\n')
print("weighted_sum", end='\n')
print(h, end='\n')
print(type(h), end='\n')
print('\n')
g = bias_first_activation(h)
print("bias_first_activation", end='\n')
print(g, end='\n')
print(type(g), end='\n')
print('\n')
w = np.add(h,g)
print("np.add(weighted_sum,bias_first_activation)", end='\n')
print(w, end='\n')
print(type(w), end='\n')
print('\n')
sig = tf.constant(w)
sigmoid = tf.sigmoid(sig)
print("tf.sigmoid(sig = tf.constant(w))", end='\n')
print(sigmoid, end='\n')
print(type(sigmoid), end='\n')
end = time.time()
runtime = end - start
print("runtime", end='\n')










