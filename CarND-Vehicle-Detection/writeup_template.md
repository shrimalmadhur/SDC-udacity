##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/image0000.png "Car Image"
[noncar]: ./output_images/image10.png "Non Car Image"
[boxes]: ./output_images/boxes.png
[pred]: ./output_images/prediction.png
[one]: ./output_images/output_1.png
[two]: ./output_images/output_2.png
[three]: ./output_images/output_3.png
[four]: ./output_images/output_4.png
[heat]: ./output_images/heatmap.png "Heatmap"
[heatpred]: ./output_images/heatmap_predict.png "Heatmap prediction"
[carhog11112]: ./output_images/car_hog_11_11_2.png
[carhog11082]: ./output_images/car_hog_11_8_2.png
[carhog09082]: ./output_images/car_hog_9_8_2.png
[carhog08082]: ./output_images/car_hog_8_8_2.png
[noncarhog11112]: ./output_images/noncar_hog_11_11_2.png
[noncarhog11082]: ./output_images/noncar_hog_11_8_2.png
[noncarhog09082]: ./output_images/noncar_hog_9_8_2.png
[noncarhog08082]: ./output_images/noncar_hog_8_8_2.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cell no `[3]` and `[4]` of the IPython notebook. I used `skimage.feature.hog` method to extract the hog features of the image  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car Image
![alt text][car]

Non Car Image
![alt_text][noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Car Image parameters (11, 11, 2)
![alt text][carhog11112]

Car Image parameters (11, 8, 2)
![alt text][carhog11082]

Car Image parameters (9, 8, 2)
![alt text][carhog09082]

Car Image parameters (8, 8, 2)
![alt text][carhog08082]

Non Car Image parameters (11, 11, 2)
![alt text][noncarhog11112]

Non Car Image parameters (11, 8, 2)
![alt text][noncarhog11082]

Non Car Image parameters (9, 8, 2)
![alt text][noncarhog09082]

Non Car Image parameters (8, 8, 2)
![alt text][noncarhog08082]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car Image 
![alt text][carhog08082]

Non Car Image
![alt text][noncarhog08082]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled with `orientations=8` , `pixel_per_cell=8` and `cell_per_block=2` because they were giving enough information about the gradients of image. I tried RGB, HSV and YCrCb and `YCrCb` was giving me the best accuracy for the classifier.
 
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklean.svm.LinearSVC` which has linear kernel. The code is in code cell `[11]`.
I used HOG features and in addition to that I also used color histogram features and spatial bin features. The code is in cell `[4]`

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in cell block `[13]`
I started with `64x64` but it was taking more time so I started increasing my window frame. I found out that `96x96` was giving good enough for information and increasing the window size will further increase my final box around the cars. But yes I did randomly also tried some the shapes like `128x128` and even `256x256`
I used 0.9 overlapping since that would give me very precise information of windows so that I can very precisely get the image window classified and would be able to threshold the images better.

![alt text][boxes]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt_text][pred]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/L3dSs38ycSo)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  The theshold I used was 1.

Here's an example result showing the heatmap from a single of frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on this frame of video:

### Here is one of the heatmaps of the images:
![alt text][heat]

### Here the resulting bounding boxes are drawn onto the thislast frame:
![alt text][heatpred]

### Here are some more detections on the test images
![alt text][one]
![alt text][two]
![alt text][three]
![alt text][four]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- One problem I got stuck into is the scale of the images. Consider it my stupidity but I was training on images which had scale from 0-1 and the frame from video was 0-255.
- Initially I was trying to use RGB for a lot of time. I tried HSV but accuracy didn't increase. After some time I tried YCrCb with all 3 hog channels and my accuracy increased from 96% to 99%
- One thing was detecting hog on every window. I could've implemented a more efficient version of detecting hog features by the method mentioned in tutorials.
- I think it might fail in the case of bumpy roads since I have clipped off the top half of image for removing false detection. It can also fail at night when ther are multiple lights and shape of car appears pretty different.


