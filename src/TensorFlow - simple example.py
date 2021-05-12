#!/usr/bin/env python
# coding: utf-8

# In[32]:


import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime


# In[33]:


# Define constant values
a = tf.constant(3, dtype=np.float32)
b = tf.constant(0, dtype=np.float32)


# In[34]:


# Define the mathematical formula
@tf.function
def model_fun(a, b):

    F = tf.math.multiply(tf.math.add(a,b), tf.exp(b))

    
    return F

Result = model_fun(a, b)
print (Result)

