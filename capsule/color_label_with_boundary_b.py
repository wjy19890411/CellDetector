import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set(color_codes=True)

def isboundary(pxl, i, j):
    if i == 0:
        if j == 0:
            if pxl[i][j] != pxl[i][j+1]:
                return True
            if pxl[i][j] != pxl[i+1][j+1]:
                return True
            if pxl[i][j] != pxl[i+1][j]:
                return True
            return False
        else:
            if j < pxl.shape[1]-1:
                if pxl[i][j] != pxl[i][j+1]:
                    return True
                if pxl[i][j] != pxl[i+1][j+1]:
                    return True
                if pxl[i][j] != pxl[i+1][j]:
                    return True
                if pxl[i][j] != pxl[i+1][j-1]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                return False
            if j == pxl.shape[1]-1:
                if pxl[i][j] != pxl[i+1][j]:
                    return True
                if pxl[i][j] != pxl[i+1][j-1]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                return False
    if i > 0 and i < pxl.shape[0]-1:
        if j == 0:
            if pxl[i][j] != pxl[i-1][j]:
                return True
            if pxl[i][j] != pxl[i-1][j+1]:
                return True
            if pxl[i][j] != pxl[i][j+1]:
                return True
            if pxl[i][j] != pxl[i+1][j+1]:
                return True
            if pxl[i][j] != pxl[i+1][j]:
                return True
            return False
        else:
            if j < pxl.shape[1]-1:
                if pxl[i][j] != pxl[i-1][j]:
                    return True
                if pxl[i][j] != pxl[i-1][j+1]:
                    return True
                if pxl[i][j] != pxl[i][j+1]:
                    return True
                if pxl[i][j] != pxl[i+1][j+1]:
                    return True
                if pxl[i][j] != pxl[i+1][j]:
                    return True
                if pxl[i][j] != pxl[i+1][j-1]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                if pxl[i][j] != pxl[i-1][j-1]:
                    return True
                return False
            if j == pxl.shape[1]-1:
                if pxl[i][j] != pxl[i+1][j]:
                    return True
                if pxl[i][j] != pxl[i+1][j-1]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                if pxl[i][j] != pxl[i-1][j-1]:
                    return True
                if pxl[i][j] != pxl[i-1][j]:
                    return True
                return False
    if i == pxl.shape[0]-1:
        if j == 0:
            if pxl[i][j] != pxl[i-1][j]:
                return True
            if pxl[i][j] != pxl[i-1][j+1]:
                return True
            if pxl[i][j] != pxl[i][j+1]:
                return True
            return False
        else:
            if j < pxl.shape[1]-1:
                if pxl[i][j] != pxl[i-1][j]:
                    return True
                if pxl[i][j] != pxl[i-1][j+1]:
                    return True
                if pxl[i][j] != pxl[i][j+1]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                if pxl[i][j] != pxl[i-1][j-1]:
                    return True
                return False
            if j == pxl.shape[1]-1:
                if pxl[i][j] != pxl[i-1][j]:
                    return True
                if pxl[i][j] != pxl[i][j-1]:
                    return True
                if pxl[i][j] != pxl[i-1][j-1]:
                    return True
                return False

listdir = os.listdir('stage1_train')
for item in range(len(listdir)):
    test = np.load('stage1_train/'+listdir[item]+'/'+listdir[item]+'.npy')
    test_new = np.zeros(test.shape, dtype = np.int32)
    flag = False
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if isboundary(test,i,j):
                test_new[i][j] = 2
            else:
                test_new[i][j] = min(test[i][j], 1)
    np.save('stage1_train'+'/'+listdir[item]+'/'+listdir[item]+'_with_boundary.npy', test_new)
    #plt.imshow(test_new)
    #plt.axis('off')
    #plt.show()
