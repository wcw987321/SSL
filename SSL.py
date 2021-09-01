#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


k = 2 # sparsity, must be 2
d = 6 # input dimension
p = 10 # ambient dimension
pp = 200 # decoder dimension
n = 100 # number of trials
mu, sigma = 0.0, 1.0 / ((d - 1) ** 0.5) # mean and variance of vectors in A and B
nn = 100 # number of outer trials
AxBz_lst = []
x1uz_lst = []


# In[3]:


def tryAB(): # generate one pair of A and B, compute expected ||Ax-Bz|| and ||x^T1-u^TZ||
    A = np.random.normal(mu, sigma, size=(d, p)) # generate A
    A[0] = 0 # zero out first row of A
    B = np.random.normal(mu, sigma, size=(d, pp)) # generate B
    B[0] = 1 # set first row of B to be all one

    def best_z(Ax, B): # find the z that minimizes ||Ax-Bz||
        cand_z = np.zeros(pp) # candidate of z
        tmp_max = 0.0
        for i in range(pp):
            for j in range(i + 1, pp):
                tmp_vec = B[:, i] - B[:, j] # b_i - b_j
                ang_dis = np.dot(Ax, tmp_vec) / (np.linalg.norm(Ax) * np.linalg.norm(tmp_vec)) # cosine of the angle between Ax and this candidate Bz
                abs_ang_dis = abs(ang_dis)
                if (abs_ang_dis > tmp_max):
                    tmp_max = abs_ang_dis
                    c = abs(np.dot(Ax, tmp_vec) / (np.linalg.norm(tmp_vec) ** 2)) # compute the magnitude of z
                    if (ang_dis > 0):
                        cand_z = np.zeros(pp)
                        cand_z[i] = c
                        cand_z[j] = -c
                    else:
                        cand_z = np.zeros(pp)
                        cand_z[i] = -c
                        cand_z[j] = c
        #print(tmp_max)
        return cand_z

    x_lst = []
    z_lst = []
    obj = 0
    for i in range(n):
        x = np.zeros(p)
        z = np.zeros(pp)
        sparse_pos_lst = random.sample(range(p), k)
        for pos in sparse_pos_lst:
            x[pos] = np.random.normal(0, 1)
        Ax = np.dot(A, x)
        z = best_z(Ax, B)
        x_lst.append(x)
        z_lst.append(z)
        obj += np.linalg.norm(np.dot(A, x) - np.dot(B, z)) ** 2
        #print(obj)
    obj /= n
    #print(Ax)
    #print(B)
    #print(z)
    #print(x_lst)
    #print(z_lst)
    #print(A)
    #print(x)
    #print(B)
    #print(z)
    #print(Ax)
    #print(np.dot(B, z))
    #print(np.dot(Ax, np.dot(B, z)) / (np.linalg.norm(Ax) * np.linalg.norm(np.dot(B, z))))
    #print(obj)
    AxBz_lst.append(obj)

    #least square solution
    tmp_lst = []
    for x in x_lst:
        tmp_lst.append(np.sum(x))
    X = np.array(tmp_lst)
    Z = np.array(z_lst)
    u = np.linalg.lstsq(Z, X, rcond=None)[0]
    #print(u)
    #tmp_s = 0
    #for i in range(n):
    #    tmp_s += (X[i] - np.dot(Z[i], u)) ** 2
    #tmp_s /= n
    #print(tmp_s)
    #print(np.linalg.norm(X - np.dot(Z, u)) ** 2 / n)
    x1uz_lst.append(np.linalg.norm(X - np.dot(Z, u)) ** 2 / n)


# In[ ]:


for _ in range(nn):
    tryAB()
#print(AxBz_lst)
#print(x1uz_lst)
plt.xlabel('E||Ax-Bz||^2')
plt.ylabel('E||x^T1-u^Tz||^2')
plt.scatter(AxBz_lst, x1uz_lst)
plt.show()


# In[ ]:


if False:
    # gradient descent to find best u

    max_iter = 10000 # maximum number of iteration
    abs_tol = 1e-4 # absolute tolerence: the threshold of closeness to zero
    rel_tol = 1e-6 # reletive tolerence: the threshold of closeness between two iterations

    def loss(u): # loss function
        tmp_sum = 0
        for j in range(n):
            tmp_sum += (np.sum(x_lst[j]) - np.dot(u, z_lst[j])) ** 2
        tmp_sum /= n
        return tmp_sum

    def grad(u): # gradient of u
        tmp_grad = np.zeros(pp)
        for j in range(n):
            tmp_grad -= (np.sum(x_lst[j]) - np.dot(u, z_lst[j])) * z_lst[j]
        tmp_grad *= 2
        tmp_grad /= n
        return tmp_grad

    iter_count = 0 # actual number of iterations
    alpha = 0.0 # step size
    u = np.random.normal(0.0, 1.0 / (pp ** 0.5), pp)
    old_u = np.zeros(pp)
    new_u = np.copy(u)
    #print("norm of diff at init: ", np.linalg.norm(old_u - new_u))
    loss_val = loss(u)
    print("initial loss:", loss_val)

    while ((loss_val > abs_tol) and (iter_count < max_iter) and (np.linalg.norm(old_u - new_u) > rel_tol)):
        iter_count += 1
        alpha = 0.1 / (iter_count ** 0.5)
        #print("u before GD: ", u)
        #print("step size:", alpha)
        #print("gradient:", grad(u))
        u -= grad(u) * alpha
        #print("u after GD: ", u)
        old_u = np.copy(new_u)
        new_u = np.copy(u)
        #print("old u: ", old_u)
        #print("new u: ", new_u)
        #print("norm diff between old and new u: ", np.linalg.norm(old_u - new_u))
        loss_val = loss(u)
        #print("loss value: ", loss_val)

    print(loss_val)
    #print(np.linalg.norm(old_u - new_u))
    print(iter_count)
    print(u)
    #print(x)
    #print(z)


# In[ ]:




