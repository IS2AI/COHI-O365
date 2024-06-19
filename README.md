# COHI-O365
We present the most diverse in number of images/labels/classes fisheye synthetic dataset with source codes and models. As well as a benchmarking testing real dataset.

COHI-O365 is a benchmark testing dataset for object detection in hemispherical/fisheye  for field of view invariant applications. It contains 1,000 real fisheye images of 74 classes sampled from the Objects365 dataset with 20.798 object instances per image on average. The images were captured using a hemispherical camera ELP-USB8MP02G-L180 with the 2,448 by 3,264 pixel resolution and manually annotated with standard axis-aligned bounding boxes afterward. The samples of raw images from the COHI-O365 dataset are shown below 

<img src="https://github.com/IS2AI/COHI-O365/blob/main/pictures/COHI-365 Sample Images.png" width="750">

The names of sampled classes and the number of bounding boxes for each class are presented in the next figure (to be inserted).

## The Fisheye versions of Objects 365 dataset
To train object detection models for the COHI-O365 dataset, the Fisheye versions of Objects 365 dataset was generated by applying a non-linear mapping to obtain fisheye-looking images. We called it the RMFV365 dataset, comprising 5.1 million images that encompass a broad spectrum of perspectives and distortions. This dataset serves as a valuable resource for training the model in generalized object recognition tasks. The samples of raw and annotated images are illustrated below.

<img src="https://github.com/IS2AI/COHI-O365/blob/main/pictures/RMFV365 Sample Images.png" width="750">

## Download the datasets
### The COHI-O365 dataset
One can access the COHI-O365 dataset using the following Google Drive link: [COHI-O365](https://drive.google.com/file/d/18O-_tdxNE7xcd6x9yTrD6-SH8i-HIkfB/view?usp=drive_link)

### The RMFV365 dataset
The RMFV365 dataset can be downloaded with a link: (Link is coming!!!). Alternatively it can be generated from [Objects365 dataset](https://www.objects365.org/overview.html) using the python scripts in transformations directory for image transformation and coordinates directory for coordinate mapping


## Requirements
### For data preprocessing
* numpy
* PIL
* pandas

### For object detection
We used YOLOv7 to train and evaluate object detection models. All needed information can be found on their official GitHub page 
[YOLOv7](https://github.com/WongKinYiu/yolov7). 

## Pre-trained models
We trained the YOLOv7 model with 36.9 M parameters on three datasets, namely Objects365, RMFV365, and a variant of RMFV365 codenamed RMFV365-v1 and evaluated the performance of models with our benchmark testing dataset - COHI-O365.



- **YOLOv7-0**: trained on the Objects365 dataset
- **YOLOv7-T1**: trained on Objects365 dataset and fisheye images transformed using lens and camera independent fisheye transformation with parameter n = 4, codenamed RMFV365-v1
- **YOLOv7-T2**: trained on RMFV365



## Results

mAP<sub>50</sub> results are summarized in the table below.

<table>
    <thead>
        <tr>
            <th rowspan="3">S/N</th>
            <th rowspan="3">Model</th>
            <th colspan="8">Test Results (%)</th>
        </tr>
        <tr>
            <th colspan="2">Objects365</th>
            <th colspan="2">RMFV365-v1</th>
            <th colspan="2">RMFV365</th>
            <th colspan="2">COHI-365</th>
        </tr>
        <tr>
            <th>mAP50</th>
            <th>mAP50:95</th>
            <th>mAP50</th>
            <th>mAP50:95</th>
            <th>mAP50</th>
            <th>mAP50:95</th>
            <th>mAP50</th>
            <th>mAP50:95</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>FPN</td>
            <td><strong>35.5</strong></td>
            <td>22.5</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>2</td>
            <td>RetinaNet</td>
            <td>27.3</td>
            <td>18.7</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>3</td>
            <td>YOLOv5m</td>
            <td>27.3</td>
            <td>18.8</td>
            <td>22.6</td>
            <td>14.1</td>
            <td>18.7</td>
            <td>10.1</td>
            <td>40.4</td>
            <td>28.0</td>
        </tr>
        <tr>
            <td>4</td>
            <td>YOLOv7-0</td>
            <td>34.97</td>
            <td><strong>24.57</strong></td>
            <td>29.1</td>
            <td>18.3</td>
            <td>24.2</td>
            <td>13.0</td>
            <td>47.5</td>
            <td>33.5</td>
        </tr>
        <tr>
            <td>5</td>
            <td>YOLOv7-T1</td>
            <td>34.3</td>
            <td>24.0</td>
            <td>32.7</td>
            <td>22.7</td>
            <td>32.0</td>
            <td>22.0</td>
            <td>49.1</td>
            <td>34.6</td>
        </tr>
        <tr>
            <td>6</td>
            <td>YOLOv7-T2</td>
            <td>34</td>
            <td>23.1</td>
            <td><strong>32.9</strong></td>
            <td><strong>23</strong></td>
            <td><strong>33</strong></td>
            <td><strong>22.8</strong></td>
            <td><strong>49.9</strong></td>
            <td><strong>34.9</strong></td>
        </tr>
    </tbody>
</table>

**Table:** Objects recognition results on Objects365, RMFV365-v1, RMFV365, and COHI-365 Testing Sets

Pre-trained model weights can be downloaded using a Google Drive link:  (Link is Coming)


## Citation

.........................



