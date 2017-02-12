##Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration]: ./output_images/calibrated.png "Image Calibration"
[undistorted]: ./output_images/input_images.png "Distorted and corrected Image"
[maghlss]: ./output_images/mag_hls_s.png "Magnitude and HLS-S threshold"
[sobelx]: ./output_images/sobelx.png "Sobel X"
[hlsl]: ./output_images/hls_l_thresh.png "HLS-L threshold"
[combined_binary]: ./output_images/combined_binary.png "Combined Binary"
[warped]: ./output_images/warped.png "Warped Image"
[historgram]: ./output_images/histogram.png "Histogram"
[curved_lines]: ./output_images/curved_lines.png "Curved lines"
[final]: ./output_images/result.png "Final Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell code `4,5,6,7` of the IPython notebook located in "./Advance_Lane_Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calibration]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images:
I used `cv2.undistort()` for the input image and used the distortion coefficient we found out in the last step. The code for this is in the cell `[14]` and `[15]`
![alt text][undistorted]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at from code cell `[16]` to `[20]`). 
I combined 4 threshold to make the final generated image.
* Magnitude Threshold
* SobelX
* HLS S threshold
* HLS L threshold

Following are the images for those thresholds
![alt text][maghlss]

Sobelx
![alt_text][sobelx]

HLS-L
![alt_text][hlsl]

Combined Binary
![alt_text][combined_binary]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `getWarpedImage()`, which appears in code cell `[23]` on my notebook.  The `getWarpedImage()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  The src points are taken by the hough lines which I got by the code for P1 with some more parameter tuning (That code is in code cell `[21]` and `[22]`). Destination points: the bottom points were the same as source and the top points were hardcoded with x component of image multiplied by 0.3.


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I plotted a historgram to find base of the peaks to find the two lane lines in the warped image. I used `scipy.signal.find_peaks_cwt` function for it (Code cells line `[27]`). The resulting histogram was:
![alt text][histogram]

Then I used a shifting algorithm (window size - 100) to find the pixels from the base points found by the above histogram.
After having the pixel values I tried to fit those pixel on a 2nd order polynomial by using `numpy.polyfit` function. It gave me the 3 parameters of the equation. The code cell for this are `[30], [32], [33]`
Here is the image of what I formed:
![alt_text][curved_lines]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Now I find the radius of curvature in the code cells `[36]`
The lane shift was found by this piece of code (Sorry I forgot to make it a function) But this code is in the code cell `[39]`:
 ```
 y_max_l = np.max(left_pixels[1])
 
 line_fitx_l = left_line_curve[0][0]*y_max_l**2 + left_line_curve[0][1]*y_max_l + left_line_curve[0][2]
 
 y_max_r = np.max(right_pixels[1])
 
 line_fitx_r = right_line_curve[0][0]*y_max_r**2 + right_line_curve[0][1]*y_max_r + right_line_curve[0][2]
 
 lane_center = (line_fitx_l + line_fitx_r)/2
 
 actual_center = image.shape[1]/2
 
 diff = lane_center - actual_center
 
 diff = diff*3.7/700
```

where `left_pixels` and `right_pixels` are the pixel values of left and right lines. `left_line_curve[0]` and `right_line_curve[0]` are the parameters of the equations of the second order polynomials.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell `[38]`, `[39]` and `[40]` and the function name is `process_image_advanced`.  Here is an example of my result on a test image:

![alt text][final]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/6KZ_lE9Z3L4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are several problems I faced:
- Initially I struggled in tuning thresholds and what all method to use to create binary image.
- Even in hough lines I spend some time tuning my parameters.
- And I faced several problems in how to find pixels and stuff. 

So I tried the challenge video and instantly saw where my techniques would fail. I saw that the shadow from the divider was confusing the algorithm.  Also the road had a half dark patch and half light patch. That was screwing up the pixel finding.
A better implementation might be to use a better lane finding algorithm. Maybe in addition of hough lines we could use another algorithm. Also I am not sure if it can be used as a method but we know the width of the lane beforehand. maybe we can leverage that in lane finding somehow.
Can left and right camera images help??
