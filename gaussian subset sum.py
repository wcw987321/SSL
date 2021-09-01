#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from matplotlib import pyplot as plt


# In[2]:


# parameter setting
p = 150 # decoder size
n = 100 # number of experiments

ans_lst = [] # list storing answers


# In[ ]:


for _ in range(n):
    lst = np.random.normal(0, 1, p)
    ans = 100.0
    for i in range(p - 2):
        for j in range(i + 1, p - 1):
            for k in range(j + 1, p):
                ans = min(ans, abs(lst[i] + lst[j] + lst[k]))
    ans_lst.append(ans)
ans_lst = sorted(ans_lst)
ans_lst = [math.log(y,10) for y in ans_lst]


# In[ ]:


plt.plot(ans_lst)
plt.show()
print(ans_lst[len(ans_lst) * 4 // 5])


# In[ ]:


p_lst = []
n = 100
range_max = 110
step_size = 10
for p in range(10, range_max, step_size):
    ans_lst = []
    for _ in range(n):
        lst = np.random.normal(0, 1, p)
        ans = 100.0
        for i in range(p - 3):
            for j in range(i + 1, p - 2):
                for k in range(j + 1, p - 1):
                    for l in range(k + 1, p):
                        ans = min(ans, abs(lst[i] + lst[j] + lst[k] + lst[l]))
        ans_lst.append(ans)
    ans_lst = sorted(ans_lst)
    #ans_lst = [math.log(y,10) for y in ans_lst]
    p_lst.append(ans_lst)


# In[3]:


plt_lst = []
for p in range(len(p_lst)):
    plt_lst.append(p_lst[p][n * 1 // 10])
plt.plot(range(10, range_max, step_size), plt_lst, label = '10%')
plt_lst = []
for p in range(len(p_lst)):
    plt_lst.append(p_lst[p][n * 1 // 4])
plt.plot(range(10, range_max, step_size), plt_lst, label = '25%')
plt_lst = []
for p in range(len(p_lst)):
    plt_lst.append(p_lst[p][n * 1 // 2])
#plt_lst = [math.log(y, 10) for y in plt_lst]
plt.plot(range(10, range_max, step_size), plt_lst, label = '50%')
plt_lst = []
for p in range(len(p_lst)):
    plt_lst.append(p_lst[p][n * 3 // 4])
plt.plot(range(10, range_max, step_size), plt_lst, label = '75%')
plt_lst = []
for p in range(len(p_lst)):
    plt_lst.append(p_lst[p][n * 9 // 10])
plt.plot(range(10, range_max, step_size), plt_lst, label = '90%')
plt.xlabel('p\'')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.show()


# In[ ]:




