import src.dataset.bps_dataset as bps_dataset
import boto3
import cv2
from botocore.config import Config
from botocore import UNSIGNED
import numpy as np
from matplotlib import pyplot as plt
from src.dataset.augmentation import NormalizeBPS

if __name__ == '__main__':
    csv_file = 'meta.csv'
    csv_dir = 'Microscopy/train'
    bucket_name = 'nasa-bps-training-data'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))    

    dataset = bps_dataset.BPSMouseDataset(csv_file, csv_dir, s3_client, bucket_name, file_on_prem = False)
    norm = NormalizeBPS()

#    for i in range(500):
       #data = dataset[i][0]
    #   cv2.imshow('image', data)
    #   cv2.waitKey(0)
       
      #  print(data)
        #data8 = (data/256).astype('uint8')
        #temp = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
       # print(temp)
       # dataNorm = norm(data)
        #print(dataNorm)
       # temp = cv2.cvtColor(dataNorm, cv2.COLOR_BGR2GRAY)
        
        #ret, thresh = cv2.threshold(temp,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #cv2.imshow('image', thresh)
        #cv2.waitKey(0)

        # # noise removal
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # # sure background area
        # sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # # Finding sure foreground area

        # opening8 = (opening/256).astype('uint8')

        # dist_transform = cv2.distanceTransform(opening8,cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg,sure_fg)

        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)

        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers+1

        # # Now, mark the region of unknown with zero
        # markers[unknown==255] = 0

        # markers = cv2.watershed(data,markers)
        # data[markers == -1] = [255,0,0]

        # cv2.imshow("image", data)
        # cv2.waitKey(0)

    
    import os
    #for j in range(50):
    #    os.mkdir(f"pics_{j}")
    #    path = f"pics_{j}"

    #for i in range(50):
        #data = dataset[16][0]
        #initial_path = os.path.join(f"try\\initial.png")
        #cv2.imwrite(initial_path, data)


        #dataNorm = norm(data) 
        #temp = cv2.cvtColor(dataNorm, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(temp,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

       #augment_path = os.path.join(f"try\\{i}.png")
       # cv2.imwrite(augment_path, thresh)

        # #noise removal
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # # sure background area
        # sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg,sure_fg)

        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)

        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers+1

        # # Now, mark the region of unknown with zero
        # markers[unknown==255] = 0

        # markers = cv2.watershed(data,markers)
        # data[markers == -1] = [255,0,0]

        # cv2.imshow('image', data)
        # cv2.waitKey(0)



    for i in range(100):
        data = dataset[i][0]
        cv2.imwrite(f'fixed\\{i}_original.png', data)


        dataNorm = norm(data)
        temp = cv2.cvtColor(dataNorm, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(temp,100,255, cv2.THRESH_BINARY)
        #cv2.imshow('i', thresh)
        #cv2.waitKey(0)
        #cv2.imwrite(f'fixed\\t_0.05_{i}.png', thresh)


        # noise removal
        kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(thresh,kernel,iterations=1)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

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

        cv2.imwrite(f'fixed\\{i}_watershed.png', data)

    # data = dataset[48][0]
    # dataNorm = norm(data)
    # temp = cv2.cvtColor(dataNorm, cv2.COLOR_BGR2GRAY)
            
    # ret, thresh = cv2.threshold(temp,0,255,16)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
