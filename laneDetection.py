import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import camera_mod, line_mod
global camera, l_line, r_line


def sanityCheck(line, fit, x, line_ref):
    
    dist_check = False
    poly_check = False
    
    # Line close to previous
    dist_check = np.amax(np.abs(line.bestx-x))< 150
    
    # Polynom close to previous
    k0, k1 = 10.0, 5.0

    diff = [abs((fit[i]-line.best_fit[i])/line.best_fit[i]) for i in range(3)]
    Small0 = (abs(fit[0]) < 2e-3)
    Small1 = (abs(fit[1]) < 1e-1) 
    
    if ((diff[0]<k0) | Small0) and ((diff[1]<k1) | Small1):
        poly_check=True
    else:
        poly_check = False
    
    sanity = (dist_check & poly_check)
    
    if sanity:
       line.sanityCount = 0
    else:
       line.sanityCount += 1
    
    if line.sanityCount > 5:
        line.detected = False
        line.filterInit = False
        print("--------- Filter re-initialized----------")
        
    return sanity


def process_video(video_in, video_out, testmode=False):
    
    from moviepy.editor import VideoFileClip
    
    if testmode:
        clip = VideoFileClip(video_in).subclip(38,50)
    else:
        clip = VideoFileClip(video_in)
       
    image_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    image_clip.write_videofile(video_out, audio=False)


def preprocess_image(img, sx_thresh=(30, 255)):

    undst = cv2.undistort(img, camera.mtx, camera.dist, None, camera.mtx) 
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(undst, camera.M, img_size, flags=cv2.INTER_AREA)
    
    # Extract wanted image color channels
    hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS).astype(np.float)
    luv = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV).astype(np.float)
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    b_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab).astype(np.float)[:,:,2]    
    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]
    l_channel = luv[:,:,0]
    
    # Thresholding
    h_binary = np.zeros_like(s_channel)
    h_binary[(h_channel > 19) & (h_channel < 25)] = 1
    
    b_thresh_min = 150
    b_thresh_max = 220
    b_binary = np.zeros_like(s_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    g_thresh_min = 215
    g_thresh_max = 255
    g_binary = np.zeros_like(s_channel)
    g_binary[(gray >= g_thresh_min) & (gray <= g_thresh_max)] = 1
   
    total_color1 = np.zeros_like(s_channel)
    total_color = np.zeros_like(s_channel)
    total_color1 = cv2.bitwise_or(h_binary, g_binary)
    total_color = cv2.bitwise_or(total_color1, b_binary)
    
    # Sobel filter in the horizontal direction
    sobelxs = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=7) # Take the derivative in x
    abs_sobelxs = np.absolute(sobelxs) # Absolute x derivative to accentuate lines away from horizontal
    msk = 0.2 * cv2.mean(s_channel)[0]
    abs_sobelxs[s_channel < msk] = 0
    scaled_sobels = np.uint8(255*abs_sobelxs/np.max(abs_sobelxs))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx[s_channel < msk] = 0
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sxbinary = np.zeros_like(scaled_sobels)
    sxbinary[(scaled_sobels >= 35) & (scaled_sobels <= 255)] = 1

    gxbinary = np.zeros_like(scaled_sobels)
    gxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Combine thresholded and gradient images    
    total_sobel = np.zeros_like(s_channel)
    total = np.zeros_like(s_channel)
    total_sobel = cv2.bitwise_or(sxbinary, gxbinary)

    total = np.zeros_like(s_channel)
    total[(total_color == 1) | (total_sobel == 1)] = 1 
    
    return total, undst
        

def process_image(img,testmode = False):
    
    # Pre-process image
    binary_warped, undist = preprocess_image(img)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, 255*binary_warped))
    
    
    if r_line.detected and l_line.detected:
                
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit = l_line.current_fit
        right_fit = r_line.current_fit
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 70
    
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]        

        # Sanity check to detect outlier
        #if np.amax(np.abs(r_line.bestx-right_fitx))< 150:
        if sanityCheck(r_line, right_fit, right_fitx, l_line):

            r_line.xAdd(right_fitx)
            r_line.detected = True           
            right_fit_m = r_line.polyMean(np.polyfit(ploty, r_line.bestx, 2))
        else:
            r_line.xAdd([])
            right_fit_m = r_line.polyMean(np.polyfit(ploty, r_line.bestx, 2))

        
        if sanityCheck(l_line, left_fit, left_fitx, r_line):

            l_line.xAdd(left_fitx)
            l_line.detected = True        
            left_fit_m = l_line.polyMean(np.polyfit(ploty, l_line.bestx, 2))
            left_fit_m = l_line.best_fit
        else:
            l_line.xAdd([])
            left_fit_m = l_line.polyMean(np.polyfit(ploty, l_line.bestx, 2))

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit_m[0]*ploty**2 + left_fit_m[1]*ploty + left_fit_m[2]
        right_fitx = right_fit_m[0]*ploty**2 + right_fit_m[1]*ploty + right_fit_m[2]
        
        l_line.allx = l_line.bestx
        r_line.allx = r_line.bestx
        l_line.ally = ploty
        r_line.ally = ploty

    else:    
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        
        if testmode:
            plt.figure(11)
            plt.imshow(binary_warped)
            plt.figure(12)
            plt.plot(histogram)
    
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 120
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
    
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        l_line.current_fit = left_fit
        r_line.current_fit = right_fit
        l_line.best_fit = left_fit
        r_line.best_fit = right_fit        
        l_line.sanityCount = 0 
        r_line.sanityCount = 0 
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        l_line.xAdd(left_fitx)
        r_line.xAdd(right_fitx)
        l_line.allx = leftx
        r_line.allx = rightx
        l_line.ally = lefty
        r_line.ally = righty
        l_line.sanityCount = 0
        r_line.sanityCount = 0
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_line.allx, l_line.ally ]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_line.allx, r_line.ally])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, camera.Minv, (img.shape[1], img.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

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
        r_line.detected = True
        

    # Calculate vehicle center
    vehicleCenter = img.shape[1]*xm_per_pix / 2
    lineLeft = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
    lineRight = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
    lineMiddle = lineLeft + (lineRight - lineLeft)/2
    diffFromVehicle = lineMiddle - vehicleCenter
    text_vc = "{:.1f} cm".format(abs(100*diffFromVehicle))
    
    font = cv2.FORMATTER_FMT_NUMPY
    fontScale = 1
    fontColor = (0,255,0)
    cv2.putText(result, "Left curvature: {:.0f} m".format(l_line.radius_of_curvature), (40, 50), font, fontScale, fontColor, 2)
    cv2.putText(result, "Right curvature: {:.0f} m".format(r_line.radius_of_curvature), (40, 100), font, fontScale, fontColor, 2)
    cv2.putText(result, "Vehicle is {} of center".format(text_vc), (40, 150), font, fontScale, fontColor, 2)

    return result


def main():
    
    global camera, l_line, r_line
    
    video_fin = "./challenge_video.mp4"
    video_fout = "./challenge_video_out.mp4"
#    video_fin = "./project_video.mp4"
#    video_fout = "./project_video_out.mp4"

    # Camera calibration
    print("Camera calibration")    
    camera = camera_mod.Camera()
    ret = camera.cameraCalibration()
    if ret:
        camera.perspectiveTransform()
        print("Camera created and calibrated")
        print("-------------------\n")
    else:
        print("No camera object initiated") 
    
    # Create line objects for left and right line
    l_line = line_mod.Line()
    r_line = line_mod.Line()

    # Process a mp4 video clip 
    process_video(video_fin, video_fout, testmode = False)
    
if __name__ == '__main__':
    main()

