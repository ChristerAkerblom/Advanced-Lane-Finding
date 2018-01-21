import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Line():
    def __init__(self, coeffN=7):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Filter initiated
        self.filterInit = False
        self.coeffN = coeffN
        self.coeffArr = []
        self.sanityCount = 0
    
    def xAdd(self, x):
        N = len(self.recent_xfitted)
        if N > self.coeffN-1 :
            self.recent_xfitted.pop()
        
        elif (len(x) == 0) and (N > 1):
            self.recent_xfitted.pop()
        
        if len(x) > 0:
            self.recent_xfitted.insert(0,x)
        
        self.xMean(x)
    
    def xMean(self, x):
        N = len(self.recent_xfitted)
        if N > 0:
            self.bestx = np.sum(self.recent_xfitted, axis=0)/N
    
    def polyMean(self, coeff):
        if self.filterInit:
            self.coeffArr.pop()

        self.coeffArr.insert(0,coeff)
        self.current_fit = coeff
        N = len(self.coeffArr)
        if N>0:        
            self.best_fit = np.sum(self.coeffArr, axis=0)/float(N)
        else:
            self.best_fit = coeff
            
        if  N >= self.coeffN: 
            self.filterInit = True
        else:
            self.filterInit = False
        
        return self.best_fit
    
def main():
    print("No test available")

if __name__ == '__main__':
    main()