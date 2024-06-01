#import libraries
import os
import math
import time
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# Labels
coco = COCO('zhiyuan_objv2_val.json')
names = [x["name"] for x in coco.loadCats(coco.getCatIds())]

patches = ['patch0', 'patch1']    #choose patches to convert!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def atan2(y, x):
    return math.atan2(y,x)

def coordinate_transform(x, y, w, h, width, heigth):
    a = 3264
    b = 2448
    c1 = 0.14
    c2 = 0.185
    f = 1000
    
    if (width < height):
        img_w = b   
        img_h = a
        w_crop = img_w - (int(c1*img_w)-10)*2
        h_crop = img_h - (int(c2*img_h)-10)*2
    else:
        img_w = a  
        img_h = b
        w_crop = img_w - (int(c2*img_w)-10)*2
        h_crop = img_h - (int(c1*img_h)-10)*2
        
    scaling_w = img_w/width
    scaling_h = img_h/height
    
    x = x*scaling_w 
    y = y*scaling_h
    w = w*scaling_w
    h = h*scaling_h
        
    #corner points of bounding box
    xtl = x
    ytl = y
    xbr = xtl + w
    ybr = ytl + h
    xtr = xbr
    ytr = ytl 
    xbl = xtl
    ybl = ybr

    #middle points
    x1 = (xtr + xtl)/2
    y1 = (ytr + ytl)/2 
    x2 = (xbl + xtl)/2
    y2 = (ybl + ytl)/2 
    x3 = (xbr + xtr)/2
    y3 = (ybr + ytr)/2 
    x4 = (xbr + xbl)/2
    y4 = (ybr + ybl)/2 
      
    coord_x_box = np.array([xtl, xbr, xtr, xbl, x1, x2, x3, x4])
    coord_y_box = np.array([ytl, ybr, ytr, ybl, y1, y2, y3, y4])
            
    #transformation    
    x = coord_x_box - img_w/2
    y = coord_y_box - img_h/2

    r = np.sqrt(x**2 + y**2) 
    vect_atan2 = np.vectorize(atan2)
    theta = vect_atan2(y,x)
                
    s1 = f*vect_atan2(r, f)
                
    x_new = s1*np.cos(theta)
    y_new = s1*np.sin(theta)

    coord_x_new_box = x_new + img_w/2
    coord_y_new_box = y_new + img_h/2
        
    #new coordinates of standard axis-aligned bbox
    xtl_new = min(coord_x_new_box)
    ytl_new = min(coord_y_new_box)

    xbr_new = max(coord_x_new_box)
    ybr_new = max(coord_y_new_box)
            
    w_new = xbr_new - xtl_new
    h_new = ybr_new - ytl_new
                      
    #find center points         
    if (img_w < img_h):
        x_centre = (xtl_new + xbr_new)/2 - (int(c1*img_w)-10)
        y_centre = (ytl_new + ybr_new)/2 - (int(c2*img_h)-10)
                
    else:
        x_centre = (xtl_new + xbr_new)/2 - (int(c2*img_w)-10)
        y_centre = (ytl_new + ybr_new)/2 - (int(c1*img_h)-10)
                
    #normalization
    x_centre = x_centre / w_crop
    y_centre = y_centre / h_crop
    w_new = w_new / w_crop
    h_new = h_new / h_crop
    
    return x_centre, y_centre, w_new, h_new





name_end = '_f'

start = time.time()

for cid, cat in enumerate(names):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    
    for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
        width, height = im["width"], im["height"]
        path = im["file_name"]
        
        patch = path.split('/')[2]
        patch_name = patch + name_end
        
        if (patch in patches):
            if (not os.path.isdir(patch_name)):
                os.mkdir(patch_name)

            im_name = path.split('/')[3]

            try:
                with open(patch_name + '/' + im_name[:-4] + name_end + '.txt', 'a') as file:
                    annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                    for a in coco.loadAnns(annIds):
                        x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                        x, y, w, h = coordinate_transform(x, y, w, h, width, height)  # transformed coordinates
                        file.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            except Exception as e:
                print(e)
                
end = time.time()

timer = (end - start)/60
print(timer)
