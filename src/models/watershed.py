import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision
import random

class Watershed(object):
    def get_mask(self, image: np.ndarray):
        """Apply watershed to images"""
        avg_intensity = self.getAverage(image)

        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        ret, thresh = cv2.threshold(bw_image,avg_intensity + 2,255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3),np.uint8)

        # sure background area
        sure_bg = cv2.dilate(thresh,kernel,iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(image,markers)
        image[markers == -1] = [255,0,0]

        return markers

    def getAverage(self, image: np.ndarray) -> int:
        sum = 0
        num_elements = 0
        
        for value in np.nditer(image):
            if value > 0: 
                sum += value
                num_elements += 1

        return int(sum / num_elements)