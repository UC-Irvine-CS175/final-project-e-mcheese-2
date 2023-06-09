'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision
import random


class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """

        normalized = np.linalg.norm(img_array)
        
        if normalized == 0:
            return img_array
        return img_array / normalized
    
class NormalizeWatershed(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """

        normalized = np.linalg.norm(img_array)
        
        if normalized == 0:
            return img_array
        
        norm_array = img_array / normalized
        norm_array *= 256
        norm_array = norm_array.astype('uint8')

        return norm_array

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.resize_height = resize_height
        self.resize_width = resize_width
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        
        resized = cv2.resize(img, dsize = (self.resize_width, self.resize_height))
        return resized


class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        
        return np.array(list(reversed(image)))



class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        rev_img = []
        for row in image:
            rev_img.append(list(reversed(row)))
        return np.array(rev_img)


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        self.rotate = rotate

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        rot_img = image

        for _ in range(int(self.rotate / 90)):
            rot_img = np.rot90(rot_img)
        
        return rot_img


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image):
        width = image.shape[1]
        height = image.shape[0]

        x_start = random.randint(0, width - self.output_width)
        y_start = random.randint(0, height - self.output_height)

        return image[y_start : y_start + self.output_height, x_start : x_start + self.output_width]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # numpy image: H x W x C
        # torch image: C x H x W

        torch_img = np.expand_dims(image, axis = 0)
        return torch.from_numpy(torch_img.copy())
class tensorToFloat(object):
    def __call__(self, imageTensor) -> np.ndarray:
        """
        Convert tensor to float
        """
        return imageTensor.float()
    
class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize
        



def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'dataset\P242_73665006707-A6_003_013_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    norm = NormalizeBPS()
    vflip = VFlipBPS()
    crop = RandomCropBPS(50, 50)
    

if __name__ == "__main__":
    main()

