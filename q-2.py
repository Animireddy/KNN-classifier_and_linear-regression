#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math

ip = 'CTTCATGTGAAAGCAGACGTAAGTCA'
spc = 'EEEEEEEEEEEEEEEEEE5IIIIIII'
probs = []

k  = 0
def func1(arg):
    k1 = k+1
    switcher = { 
        'A': 0, 
        'G': 2, 
        'T': 3, 
        'C': 1, 
    }
    return switcher.get(arg, "nothing")

cpb = [[0.25, 0.25, 0.25, 0.25], [0.05, 0, 0.95, 0], [0.4, 0.1, 0.1, 0.4]]

def func2(arg):
    k1 = k+1
    switcher = { 
        'E': 0, 
        '5': 1, 
        'I': 2,  
    }
    return switcher.get(arg, "nothing")
    
state_change_prob = [[0.9, 0.1, 0], [0, 0, 1.0], [0, 0, 0.9]]

probability = 1.0 * cpb[func2(spc[0])][func1(ip[0])]

# print(probability)

probs.append(math.log(probability))

l1 = 0
for i in range(1, len(ip)):
    l1 = l1 + 1
    probability *= (state_change_prob[func2(spc[i-1])][func2(spc[i])] * cpb[func2(spc[i])][func1(ip[i])])
    probs.append(math.log(probability))
    l1 = l1 + 1
    if spc[i]=='5':
        print(math.log(probability))
    
probability *= 0.1

lgp = math.log(probability)

probs.append(lgp)

# print(log_probability)


# In[ ]:




