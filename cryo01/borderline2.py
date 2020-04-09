from load_sample import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import ndimage
import sys



#Variables
#n = 15
area_threshold = 0.8
# sigma = 1
resolutions = (512,512,3)
#filename,dic,Y, X, classes = load_samples(image_size=resolutions)


# n_openings = 3
# n_erosions = 8
# n_dilations = 8
# save = 0


#Function
def ImageProperties(n,img_array,area_array,area_threshold,save,n_openings,n_erosions,n_dilations,filename):
    
    Img = area_array[n]

    Open = ndimage.binary_opening(Img,                       iterations = n_openings)
    Eroded = ndimage.binary_erosion(Open,                    iterations = n_erosions)
    Dilation = ndimage.binary_dilation(Eroded, mask=Open,    iterations = n_dilations)
    label_im, nb_labels = ndimage.label(Dilation)
    
    #Plots
    
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6)
    
    fig.set_figheight(10)
    fig.set_figwidth(22)
    
    ax1.set_title('Original \n'+str(filename[n]))
    im1= ax1.imshow(img_array[n], cmap = 'gist_gray')
    
    ax2.set_title('Area \n Area Threshold: '+str(area_threshold))
    im2= ax2.imshow(Img, cmap = 'gist_stern')
    
    ax3.set_title('binary opening \n # Openings: '+str(n_openings))
    im3= ax3.imshow(Open, cmap = 'gist_stern')
    
    ax4.set_title('binary erosion \n # Erosions: '+str(n_erosions))
    im4= ax4.imshow(Eroded, cmap = 'gist_stern')    
    
    ax5.set_title('binary dilation \n # Dilations: '+str(n_dilations))
    im5= ax5.imshow(Dilation, cmap = 'gist_stern')
    
    ax6.set_title('Labeled Image \n '+str(nb_labels)+' Crystals')
    im6= ax6.imshow(label_im)
    
    
    if save ==1:
       
        fig.savefig('ML//Test//image_properties//'+str(n+1)+'_'+str(filename[n]))
        plt.close(fig)
    #print('status: '+str(n+1)+'/'+str(X.shape[0]))
    
    Dilation = Dilation[np.newaxis,:]
    
    
    sys.stdout.write("\r" + '   Progress: '+str(n+1)+'/'+str(img_array.shape[0]))
    sys.stdout.flush()
    plt.close()
    
    return Dilation




# #Metrics
# sizes       = ndimage.sum(Open, label_im, range(nb_labels + 1))
# mean_vals   = ndimage.sum(Img, label_im, range(1, nb_labels + 1))

