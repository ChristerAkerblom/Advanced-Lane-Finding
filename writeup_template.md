****Advanced Lane Finding Project****

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

[image05]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image25]:  ./output_images/undistorted_test2.png ""
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image45]: ./output_images/warped_straight_lines1.png "Warp Example"
[image4]: ./output_images/straight_lines1.png "Warp Example"
[image5]: ./output_images/color_fit_lines2.png "Fit Visual"
[image6]: ./output_images/video_out.png "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function 'cameraCalibration()' in the file called `camera_mod.py`.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image05]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Original image:
![alt text][image2]

Undistorted image:
![alt text][image25]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Thresholding steps are defined in function `preprocess_image` in `lanes.py`).

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The class Camera defined in `camera_mod.py` includes the perspective transform and is calculated at initialization. I found the source and destination points by experimenting with an image with straight lines which resulted in the following source and destination points:
```python

src_pts = np.float32(((578,460),(255, 680), (1045,680), (702,460)))
dst_pts = np.float32(((255,20), (255, 710), (1015,710), (1015,20))) 

```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image45]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function `process_image()` in `laneDetection.py` identifies lane-line pixels and fit them to a second order polynom. The result is presented in the image below. Note that there is some none line pixels present in the right line fit corrupting the estimate somewhat.
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was calculated according to the formula describe in the course. A lowpass filter was applied to each line curvature since it can be assumed to be a slowly changing variable. Curvatures above 10000 m was not allowed, not to end up in problems with the filtering.

´´´

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/790 # meters per pixel in x dimension
    y_eval = result.shape[0] * ym_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    k = 0.01
    if left_curverad > 10000:left_curverad = 10000 
    if right_curverad > 10000:right_curverad = 10000 
    if (l_line.detected & r_line.detected):
        l_line.radius_of_curvature = (1.0-k) * l_line.radius_of_curvature + k * left_curverad
        r_line.radius_of_curvature = (1.0-k) * r_line.radius_of_curvature + k * right_curverad
    else:
        l_line.radius_of_curvature = left_curverad
        r_line.radius_of_curvature = right_curverad
        l_line.detected = True
´´´

I did this in lines 312 through 374 in my code in function `process_image()` in `laneDetection.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 294 through 353 in my code in the function `process_image()` in `laneDetection.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
In general the algorithm implemented used color and gradient information detect lane-line pixels that were then fitted to a second order polynom. The polynom was lowpass filtered by taking mean over five samples. An outlier detection was implemented identifying if an line deviates to much from it predecessors considering distance in x-axis and polynom parameters.

The algorithm works well for the standard scenarios but for really curvy roads a shorter horizon is probably needed and also another approach to fitting. I would have liked to look into splines. To support this adaptivity a curvy road detector is needed. 
