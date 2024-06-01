#import libraries
import os
from PIL import Image
import math
import numpy as np
import time

def atan2(y, x):
    return math.atan2(y,x)

def equidistant_projection(filename, file_path, output_dir, f, a, b, name_end):
    
    image_orig = Image.open(file_path)      
    h = image_orig.size[1]
    w = image_orig.size[0]   
    
    #resize to have the same aspect ration as images from COHI and reduce image quality degradation after non-linear mapping
    if (w<h):
        image_resized = image_orig.resize((2448, 3264), Image.BICUBIC);       
    else: #w>h w==h
        image_resized = image_orig.resize((3264, 2448), Image.BICUBIC);
              
    h = image_resized.size[1]
    w = image_resized.size[0]  
    I = np.asarray(image_resized)
         
    y = np.arange(h)
    x = np.arange(w)   
    xi, yi = np.meshgrid(x, y)
    
    xt = xi - w/2
    yt = yi - h/2
    
    r = np.sqrt(xt**2 + yt**2)
    vect_atan2 = np.vectorize(atan2)
    theta = vect_atan2(yt, xt)
    
    s1 = f*vect_atan2(r, f)
    
    x2 = s1*np.cos(theta) 
    y2 = s1*np.sin(theta)
    
    vector = np.vectorize(np.int_)
    xf = vector(x2 + w/2)
    yf = vector(y2 + h/2)
               
    if image_resized.mode == "RGB":    #RGB
        I_new = np.zeros([h, w, 3])
        
    else:    #grayscale ('L')
        I_new = np.zeros([h, w])
        
    I_new[yf, xf] = I[yi, xi]
                              
    I_new = np.uint8(I_new)
    image_transformed = Image.fromarray(I_new)
    
    #crop black borders
    
    if h>w:
        image_cropped = image_transformed.crop([int(b*w)-10, int(a*h)-10, w-int(b*w)+10, h-int(a*h)+10])
    else:
        image_cropped = image_transformed.crop([int(a*w)-10, int(b*h)-10, w-int(a*w)+10, h-int(b*h)+10])
        
    image_cropped.save(output_dir + filename[:-4] + name_end + '.jpg')








directory = 'patch0'    #change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
start = time.time()
count = 1

for filename in os.listdir(directory):

    file_path = os.path.join(directory, filename)
    print(file_path + " " + str(count))
    count = count+1

    try:
        equidistant_projection(filename, file_path, 'patch0_labels_f/', 1000, 0.185, 0.14, '_f')   #change output dir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    except Exception as e:
        print(e)

end = time.time()
timer = (end - start)/60
print(timer)
    
