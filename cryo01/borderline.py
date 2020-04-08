from load_sample import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys

resolutions = (512,512,3)
filename,dic,Y, X, classes = load_samples(image_size=resolutions)
    

def image_prop(numb,img_shape,area_threshold,thicc,proximity,quadrant_threshold,array):
    # numb : image index on array
    # img_shape : (#rows,#columns,colorchannels)
    # area_threshold : float 0 to 1 of intensity threshold to be considered in the area mapping
    # thicc : pixel thickness of the borders
    
    image = array[numb]  # Selected Image
    
    #image = image[:,:,1] + image[:,:,2] + image[:,:,0]   # If we want to join the Red,Green,Blue channels
    image =  image[:,:,2]                                 # Use only one channel
    image = image / 255
    
    image = normalize(image,norm='max')
    area = image > area_threshold

    
    #New area assignment
    import copy
    new_area = copy.deepcopy(area)
    q_range = proximity
    max_int = (q_range*2+1)**2
    q = range(-1*q_range,q_range+1)
    jornal = []
    for y in range (new_area.shape[0]-1):
        for x in range(new_area.shape[1]-1):
            
            for l1 in q:
                for l2 in q:
                    if area[y][x] == 1:
                        try: 
                            jornal.append(area[y+l1][x+l2])
                        except:
                            continue
            if sum(jornal) < (quadrant_threshold)*max_int:
                new_area[y][x] = 0
            else:
                new_area[y][x] = 1
            jornal = []
        #print(y)
    
        
    border = np.zeros((img_shape[:2]))
    for y in range(image.shape[0]-1):
        for x in range(image.shape[0]-1):
            if new_area[y][x] == new_area[y][x+1]:
                border[y][x] = 0
            else:
                z = range(-1*thicc,thicc+1)
                for o in range(len(z)):
                    for p in range(len(z)):
                        try:
                            border[y+z[p]][x+z[o]] = 1
                        except:
                            continue
                        
                        try:
                            border[y+z[o]][x+z[p]] = 1
                        except:
                            continue
    objective = np.zeros((img_shape))
    for y in range(image.shape[0]-1):
        for x in range(image.shape[0]-1):
           objective[y][x] = (10)*border[y][x] + new_area[y][x]
    objective = normalize(objective,norm='max')
    key = int(Y[numb])
    desc = dic[key]
    name = filename[key]
    return name,desc,image,area,border, objective, new_area
    
    #return new_area

# Loops all images in dataset array

for n in range(X.shape[0]):

#for n in range (0,1):
    name,desc,image,area, border,objective,new_area = image_prop(n,resolutions[:2],0.65,thicc=1,proximity=3,quadrant_threshold=0.9,array=X)
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
    
    fig.set_figheight(10)
    fig.set_figwidth(22)
    
    ax1.set_title('normal \n intensity:'+str(int(100*np.sum(image)/(image.shape[0]*image.shape[1])))+'%')
    im1= ax1.imshow(image)
    
    ax2.set_title('area \n occupancy:'+str(int(100*np.sum(area)/(image.shape[0]*image.shape[1])))+'%')
    im2= ax2.imshow(area)

    ax3.set_title('new_area \n occupancy:'+str(int(100*np.sum(new_area)/(image.shape[0]*image.shape[1])))+'%')
    im3= ax3.imshow(new_area,cmap='viridis')    
    
    ax4.set_title('border')
    im4= ax4.imshow(border)
    
    ax5.set_title('objective')
    im5= ax5.imshow(objective,cmap='viridis')    
        
    fig.savefig('ML//Test//image_properties//'+str(n+1)+'_'+str(desc)+'.PNG')
    plt.close(fig)
    #print('status: '+str(n+1)+'/'+str(X.shape[0]))
    sys.stdout.write("\r" + 'status: '+str(n+1)+'/'+str(X.shape[0]))
    sys.stdout.flush()

#center = copy.deepcopy(new_area)


"""
img_shape = resolutions
area_threshold = 0.9
thicc = 1
proximity = 3
quadrant_threshold = 0.9
array = X

import sys

for numb in range(X.shape[0]):
    if numb ==0:
        newX = image_prop(numb,img_shape,area_threshold,thicc,proximity,quadrant_threshold,array)
    else:
        current = image_prop(numb,img_shape,area_threshold,thicc,proximity,quadrant_threshold,array)
        newX = np.concatenate((newX,current),axis=0)
    sys.stdout.write("\r" + 'Progress: '+str(numb+1)+'/'+str(X.shape[0]))
    sys.stdout.flush()
    
import h5py
h5f = h5py.File('new_area_data.h5', 'w')
h5f.create_dataset('X_new', shape= (X.shape[0],resolutions[0],resolutions[1],1),data = newX)
h5f.close()
"""