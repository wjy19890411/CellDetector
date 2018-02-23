import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import csv
from matplotlib import cm
from os import listdir

def main():
    print('hello')
    print('bye')
    pass

if __name__ == '__main__':
    main()

sns.set(color_codes = True)

colors = [[1., 0., 0.], [0. ,1. , 0.], [0., 0., 1.], [1., 1., 0.]]

f = open('stage1_train_labels.csv', newline='', encoding='utf-8')
reader = csv.reader(f)
data = [] # store the data as list element by each row
for row in reader:
    data.append(row)

images_dir = listdir('stage1_train')
images_size = {}
for directory in images_dir:
    image = plt.imread('stage1_train/'+directory+'/images/'+directory+'.png')
    images_size[directory] = [image.shape[0], image.shape[1]]


def str2int(run_length):  # run_length.shape as ['3423 12 13424 3 123 41']
    tx = []
    tp = str('')
    i = 0
    while(i < len(run_length)):
        tp = str('')
        while(i < len(run_length) and run_length[i] != ' '):
            tp = tp + run_length[i]
            i = i + 1
        i = i + 1
        tx.append(tp)
    tx_int = []
    for i in range(len(tx)):
        tx_int.append(int(tx[i]))
    return tx_int

def find_center(run, height):   # run.shape as [3423, 12, 13424, 3, 123, 41]
    mass_sum = np.array([0., 0.])
    for i in range(len(run)//2):
        temp = np.array([run[i*2+1]*(run[i*2]//height), run[i*2+1]*np.mod(run[i*2], height) + run[i*2+1]*(run[i*2+1]-1)/2.])
        mass_sum = mass_sum + temp
    center = mass_sum / sum(run[1::2])
    return center

#def closet_3(cen): # cen.shape as [np.array([123.1, 81.3]), np.array([92.1, 63.3]), ..., np.array([172.1, 98.2])]


def get_color(cen): #cen.shape as [np.array([123.1, 81.3]), np.array([92.1, 63.3]), ..., np.array([172.1, 98.2])]
    cen_color = [1, 2, 3, 4]
    if len(cen) < 5:
        return cen_color[:len(cen)]
    if len(cen) > 4:
        for i in range(4, len(cen)):
            cen_i_dist = [sum(np.square(cen[i]-k)) for k in cen[:i]]
            #print('cen_i_dist before del: ', cen_i_dist)
            #print('cen_color_i is :       ', cen_color[:i])
            ind = cen_i_dist.index(min(cen_i_dist))
            s1 = cen_color[ind]
            cen_i_dist[ind] = np.inf
            ind = cen_i_dist.index(min(cen_i_dist))
            s2 = cen_color[ind]
            cen_i_dist[ind] = np.inf
            ind = cen_i_dist.index(min(cen_i_dist))
            s3 = cen_color[ind]
            cen_i_dist[ind] = np.inf
            #print('cen_i_dist after del : ', cen_i_dist)
            #print('s1, s2, s3,:', [s1, s2, s3], '\n')
            if 1 not in [s1, s2, s3]:
                cen_color.append(1)
                continue
            if 2 not in [s1, s2, s3]:
                cen_color.append(2)
                continue
            if 3 not in [s1, s2, s3]:
                cen_color.append(3)
                continue
            cen_color.append(4)
    return cen_color

center = []
color_choice = []
for i in range(1, len(data)):
    center.append([data[i][0], find_center(str2int(data[i][1]), height = images_size[data[i][0]][0])])


def plotsave(perpic, i, height, width):
    # perpic.shape as ['131 12 4352 23 1234 3', ..., '...']
    # wrong: perpic.shape as [['adf12ljkaofjd1', '131 12 4352 23 1234 3'], ..., ['adf12ljkaofjd1', '4352 23 1234 3']]
    # i is the index of last perpic's item in data
    sub_num = len(perpic)
    flat_list = [item for sublist in center[i-sub_num:i] for item in sublist]
    spot = flat_list[1::2]
    spot_color = get_color(spot)  # spot_color.shape as [1, 2, 4, 2, 4, 3, 1, 1]
    mask_colored = np.zeros(shape=[height, width, 3], dtype=np.float32)
    mask_labeled = np.zeros(shape=[height, width], dtype=np.int32)
    for k in range(sub_num):
        pixel = str2int(perpic[k])
        for ki in range(len(pixel)//2):
            start = [np.mod(pixel[ki*2], height), pixel[ki*2]//height]
            #print('start is:', start, ' i is :', i)
            length = pixel[ki*2 + 1]
            if start[0] + length <= height:
                mask_colored[start[0]:start[0]+length, start[1], :] = colors[spot_color[k]-1]
                mask_labeled[start[0]:start[0]+length, start[1]] = spot_color[k]
            else:
                mask_colored[start[0]:, start[1], :] = colors[spot_color[k]-1]
                mask_labeled[start[0]:, start[1]] = spot_color[k]
                #mask_colored[0:length-height+start[0], start[1], :] = colors[spot_color[k]-1]
    return mask_colored, mask_labeled


perpic = [data[1][1]]
for i in range(2, len(data)):
    if data[i][0] == data[i-1][0]:
        perpic.append(data[i][1])
        if i == len(data)-1:
            mask, label = plotsave(perpic, i, height = images_size[data[i][0]][0], width = images_size[data[i][0]][1])
            np.save('stage1_train/'+data[i][0]+'/'+data[i][0]+'.npy', label)
            plt.figure(data[i][0])
            plt.imshow(mask)
            plt.imsave('stage1_train/'+data[i][0]+'/'+data[i][0]+'.png', mask)
            plt.axis('off')
            plt.show()
    else:
        mask, label = plotsave(perpic, i-1, height = images_size[data[i-1][0]][0], width = images_size[data[i-1][0]][1])
        #plt.figure(data[i-1][0])
        #plt.imshow(mask)
        np.save('stage1_train/'+data[i-1][0]+'/'+data[i-1][0]+'.npy', label)
        plt.imsave('stage1_train/'+data[i-1][0]+'/'+data[i-1][0]+'.png', mask)
        #plt.axis('off')
        #plt.show()
        perpic = [data[i][1]]




'''
perpic = []
isnew = True
for i in range(1, len(data)):
    if isnew:
        perpic.append(data[i][1])
        isnew = False
    else:
        if data[i][1] == data[i-1][1]:
            perpic.append(data[i][1])
            if i == len(data)-1:
                #plotperpic[]
        else:
            #plotperpic[]
            perpic = [data[i][1]
            isnew = True



#    while(i>1 and i<len(data) and data[i][0]==data[i-1][0])
'''

'''
plt.figure()
fig1 = plt.subplot(121)
for i in range(27):
    plt.plot(center[i][1][0], -center[i][1][1],'ro')

fig1.set_xlim([0, 256])
fig1.set_ylim([0-256, 256-256])
fig1.set_aspect('equal',adjustable='box')

fig2 = plt.subplot(122)
c1 = center[:27]
flat_list = [item for sublist in c1 for item in sublist]
c1_f = flat_list[1::2]
c1_color = get_color(c1_f)
for i in range(27):
    plt.plot(center[i][1][0], -center[i][1][1], 'o',  color = colors[c1_color[i]] )

fig2.set_xlim([0, 256])
fig2.set_ylim([0-256, 256-256])
fig2.set_aspect('equal',adjustable='box')
plt.show()
'''

#pdb.set_trace()
