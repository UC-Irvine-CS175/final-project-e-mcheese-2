import sys
sys.path.append(".")

import src.dataset.bps_dataset as bps_dataset
import boto3
import cv2
from botocore.config import Config
from botocore import UNSIGNED
import numpy as np
from matplotlib import pyplot as plt
from src.dataset.augmentation import NormalizeBPS, ApplyWatershed

def test_watershed():
    for i in range(100):
        data = dataset[i][0]
        data_norm = norm(data)
        avg_intensity = my_watershed.getAverage(data_norm)
                
        #cv2.imwrite(f'test_output\\{i}.png', data)


        temp = cv2.cvtColor(data_norm, cv2.COLOR_BGR2GRAY)  
        
        ret, thresh = cv2.threshold(temp,avg_intensity + 2,255, cv2.THRESH_BINARY)
        #cv2.imwrite(f'test_output\\{i}_thresh.png', thresh)

        # noise removal
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

        markers = cv2.watershed(data,markers)
        data[markers == -1] = [255,0,0]

        cv2.imwrite(f'test_output\\{i}_watershed.png', data)
        

if __name__ == '__main__':
    csv_file = 'meta.csv'
    csv_dir = 'Microscopy/train'
    bucket_name = 'nasa-bps-training-data'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))    

    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, file_on_prem = False)
    norm = NormalizeBPS()
    my_watershed = ApplyWatershed()

    test_watershed()
