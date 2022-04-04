import numpy as np
# import matplotlib.pyplot as plt
import math
def vector(p1,p2):
    v = [p1[0] - p2[0], p1[1] - p2[1]]
    return v
def vector_set(polygon1, polygon2):
    vectorset = []
    for i in range(len(polygon1)):
        if i < len(polygon1)-1:
            vectorset.append(vector(polygon1[i],polygon1[i+1]))
        else:
            vectorset.append(vector(polygon1[-1],polygon1[0]))
    for i in range(len(polygon2)):
        if i < len(polygon2)-1:
            vectorset.append(vector(polygon2[i+1],polygon2[i]))
        else:
            vectorset.append(vector(polygon2[0],polygon2[-1]))
    return vectorset
def calangle(v1, v2):
    # v1旋转到v2，逆时针为正，顺时针为负
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
    if rho < 0:
        return -theta
    else:
        return theta
def first_step(s):
    s1 = s
    a = abs(calangle(s[0],s[1]))
    min_angle = calangle(s[0], s[1])
    index = 1
    for i in range(2,len(s)):
        if a > abs(calangle(s[0],s[i])):
            a = abs(calangle(s[0],s[i]))
            index = i
            min_angle = calangle(s[0],s[i])
    t = s[1]
    s1[1] = s1[index]
    s1[index] = t
    return s1,min_angle
def clockwise_seq(s):
    s1 = s
    for k in range(2,len(s1)):
        for i in range(k, len(s1)):
            if calangle(s1[k-1], s1[i]) >= 0:
                t = s1[k]
                s1[k] = s1[i]
                s1[i] = t
        for j in range(k+1, len(s)):
            if 0 <= calangle(s1[k - 1], s1[j]) < calangle(s1[k - 1], s1[k]):
                    t = s1[k]
                    s1[k] = s1[j]
                    s1[j] = t
    return s1
def counterclockwise_seq(s):
    s1 = s
    for k in range(2,len(s1)):
        for i in range(k, len(s1)):
            if calangle(s1[k-1], s1[i]) <= 0:
                t = s1[k]
                s1[k] = s1[i]
                s1[i] = t
        for j in range(k+1, len(s)):
            if 0 >= calangle(s1[k - 1], s1[j]) > calangle(s1[k - 1], s1[k]):
                    t = s1[k]
                    s1[k] = s1[j]
                    s1[j] = t
    return s1
def get_seq(s):
    s1 = first_step(s)
    min_angle = s1[1]
    s2 = s1[0]
    if min_angle == 0:
        del s2[1]
        s1 = first_step(s2)
        min_angle = s1[1]
        s2 = s1[0]
        if min_angle == 0:
            print("输入多边形有误")
            return 0
        elif min_angle > 0:
            s2 = clockwise_seq(s2)
            s2 .insert(0,s1[0][0])
            return s2
        else:
            s2 = counterclockwise_seq(s2)
            s2.insert(0, s1[0][0])
            return s2
    elif min_angle > 0:
        return clockwise_seq(s2)
    else:
        return counterclockwise_seq(s2)
def get_nfp(s):
    s1 = []
    s2 = []
    nfp = []
    for i in range(len(s)):
        s1.append(s[i][0])
        s2.append(s[i][1])
        x = sum(s1[:i+1])
        y = sum(s2[:i+1])
        nfp.append([x,y])
    return nfp
polygon1 = [[0,0],[1,1],[1,0]]
polygon2 = [[0,0],[0,1],[1,0]]
vs = vector_set(polygon1,polygon2)
print(vs)
seq = get_seq(vs)
print(seq)
nfp = get_nfp(seq)
print(nfp)

