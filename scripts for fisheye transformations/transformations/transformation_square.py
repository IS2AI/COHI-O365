#import libraries
import os
from PIL import Image
import math
import numpy as np
import time

def atan2(y, x):
    return math.atan2(y,x)

def square(filename, file_path, output_dir, n, scaling_factor, c, b, name_end):
    
    image_orig = Image.open(file_path)      
    h1 = image_orig.size[1]
    w1 = image_orig.size[0]

    if (h1 > 8000) or (w1 > 8000):
        s = 0.25
    elif (h1 > 4000) or (w1 > 4000): 
        s = 0.5
    elif (h1 > 2000) or (w1 > 2000): 
        s = 1
    elif (h1 > 1000) or (w1 > 1000): 
        s = 2
    else:
        s = 4
    
    #resize to have the same aspect ration as images from COHI and reduce image quality degradation after non-linear mapping
    if (w1<h1):
        image_resized = image_orig.resize((int(s*w1), int(s*int(w1*scaling_factor))), Image.BICUBIC);     
    else: #w>h w==h
        image_resized = image_orig.resize((int(s*int(h1*scaling_factor)), int(s*h1)), Image.BICUBIC);
              
    h = image_resized.size[1]
    w = image_resized.size[0]  
    image = np.asarray(image_resized)
        
    if image_resized.mode == "RGB":    #RGB
        #convert to square
        if (h < w):
            I = np.zeros([w, w, 3])
            a = int((w - h)/2)
            I[a:(a+h), :, :] = image[:,:,:]
        elif (h > w):
            I = np.zeros([h, h, 3])
            a = int((h - w)/2)
            I[:, a:(a+w), :] = image[:,:,:]
        else:
            I = image
        
        h = I.shape[1]
        w = I.shape[0]
        I_new = np.zeros([h, w, 3])
        
    else:    #grayscale ('L')
        #convert to square
        if (h < w):
            I = np.zeros([w, w])
            a = int((w - h)/2)
            I[a:(a+h), :] = image[:,:]
        elif (h > w):
            I = np.zeros([h, h])
            a = int((h - w)/2)
            I[:, a:(a+w)] = image[:,:]
        else:
            I = image
        
        h = I.shape[1]
        w = I.shape[0]
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
    
    if h1>w1:
        image_cropped = image_transformed.crop([int(b*w)-10, int(c*h)-10, w-int(b*w)+10, h-int(c*h)+10])
    else:
        image_cropped = image_transformed.crop([int(c*w)-10, int(b*h)-10, w-int(c*w)+10, h-int(b*h)+10])
    
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
        square(filename, file_path, 'patch0_labels_square/', 4, scaling_factor, 0.11, 0.17, '_square')   #change output dir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    except Exception as e:
        print(e)

end = time.time()
timer = (end - start)/60
print(timer)
    
