#import libraries
import os
from PIL import Image
import math
import numpy as np
import time

def atan2(y, x):
    return math.atan2(y,x)

def fisheye_transform(filename, file_path, output_dir, n, scaling_factor, a, name_end):
    
    image_orig = Image.open(file_path)      
    h = image_orig.size[1]
    w = image_orig.size[0]

    if (h > 8000) or (w > 8000):
        s = 0.25
    elif (h > 4000) or (w > 4000): 
        s = 0.5
    elif (h > 2000) or (w > 2000): 
        s = 1
    elif (h > 1000) or (w > 1000): 
        s = 2
    else:
        s = 4
    
    #resize to have the same aspect ration as images from COHI and reduce image quality degradation after non-linear mapping
    if (w<h):
        image_resized = image_orig.resize((int(s*w), int(s*int(w*scaling_factor))), Image.BICUBIC);     
    else: #w>h w==h
        image_resized = image_orig.resize((int(s*int(h*scaling_factor)), int(s*h)), Image.BICUBIC);
              
    h = image_resized.size[1]
    w = image_resized.size[0]  
    I = np.asarray(image_resized)
        
    if image_resized.mode == "RGB":    #RGB
        
        I_new = np.zeros([h, w, 3])
        
    else:    #grayscale ('L')
        
        I_new = np.zeros([h, w])
        
    y = np.arange(h)
    x = np.arange(w)   
    xi, yi = np.meshgrid(x, y)
    
    #normalize between -1 and +1
    x = 2*xi/w - 1
    y = 2*yi/h - 1
    
    x1 = x*np.sqrt(1 - y**2/2)
    y1 = y*np.sqrt(1 - x**2/2)

    r = np.sqrt(x1**2 + y1**2)

    x2 = x1*np.exp(-r**2/n)
    y2 = y1*np.exp(-r**2/n)
    
    vector = np.vectorize(np.int_)
    xf = vector(w*(x2+1)/2)
    yf = vector(h*(y2+1)/2)
        
    I_new[yf, xf] = I[yi, xi]
                        
    I_new = np.uint8(I_new)
    image_transformed = Image.fromarray(I_new)
    
    #crop black borders
    image_cropped = image_transformed.crop([int(a*w)-2, int(a*h)-2, w-int(a*w)+2, h-int(a*h)+2])
    
    image_cropped.save(output_dir + filename[:-4] + name_end + '.jpg')  





a = 3264;
b = 2448;

scaling_factor = a/b;    #aspect ratio of images in COHI

directory = 'patch0'    #change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
start = time.time()
count = 1

for filename in os.listdir(directory):

    file_path = os.path.join(directory, filename)
    print(file_path + " " + str(count))
    count = count+1

    try:
        fisheye_transform(filename, file_path, 'patch0_labels_7/', 7, scaling_factor, 0.065, '_7')   #change output dir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    except Exception as e:
        print(e)

end = time.time()
timer = (end - start)/60
print(timer)
    
