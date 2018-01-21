import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

global camera, l_line, r_line

class Camera:
    def __init__(self):
        self.calibrated = False
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []
        self.M = []
        self.Minv = []

    def cameraCalibration(self, nx=9,ny=6):
    
        objpoints = [] # real world object points in 3D
        imgpoints = [] # image points in 2D
        
        # Prepare object points
        objp = np.zeros((nx*ny,3), dtype=np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
        # Make a list of calibration images
        images = glob.glob('./camera_cal/*.jpg')
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
    
            # If found, add object points, image points (after refining them)
            if ret > 0:
                #print(fname)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
    
                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (nx,ny), corners2,ret)
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret >  0:
            self.calibrated = True
            self.mtx = mtx
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
        
        return ret, mtx, dist, rvecs, tvecs

    def perspectiveTransform(self):
        src_pts = np.float32(((578,460),(255, 680), (1045,680), (702,460)))
        dst_pts = np.float32(((255,20), (255, 710), (1015,710), (1015,20)))        
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
      
def main():
    global camera, l_line, r_line

    # Plot corners in image transform polygon
    fig = 0
    plt.figure(fig)    
    img=plt.imread("./test_images/straight_lines1.jpg")
    plt.imshow(img)
    plt.plot(255,680,"*")
    plt.plot(1045,680,"*")
    plt.plot(573,460,"*")
    plt.plot(705,460,"*")

    # Camera calibration
    print("Camera calibration")    
    camera = Camera()
    ret = camera.cameraCalibration()
    if ret:
        camera.perspectiveTransform()
        print("Camera created and calibrated")
        print("-------------------")
    else:
        print("No camera object initiated") 
    
    # Test to undistort images
    fname = "./camera_cal/calibration1.jpg"
    img = cv2.imread(fname)
    fig += 1
    plt.figure(fig)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    dst = cv2.undistort(img, camera.mtx, camera.dist, None, camera.mtx)
    fig += 1
    plt.figure(fig)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.imsave('./examples/undistort_output.png', dst)    
    
    fname = "./test_images/test6.jpg"
    img = cv2.imread(fname)
    dst = cv2.undistort(img, camera.mtx, camera.dist, None, camera.mtx)
    fig += 1
    plt.figure(fig)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imsave('//gradient.png', dst)
    fig += 1
    plt.figure(fig)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    main()