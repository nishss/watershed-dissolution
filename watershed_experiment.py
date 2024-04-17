import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image

def show_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def binarize_image(image):
    binary_image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv.THRESH_BINARY_INV, 11, 2)
    show_image(cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR), "Binary Image")
    return binary_image

if __name__ == "__main__":
    
    image = cv.imread("C:\\Users\\nishss\\Documents\\InFocus\\C\\20240404123108203.tif", flags=cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    

    #binary conversion here
    thresh = binarize_image(gray)
    new_array = np.zeros(thresh.shape)
    new_array[thresh == 0] = 255 
    thresh = new_array
    plt.figure()
    plt.imshow(thresh)
    plt.show()



    #now remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN, kernel, iterations = 2)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    plt.figure()
    plt.imshow(sure_bg)
    plt.show()
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    plt.figure()
    plt.imshow(sure_fg)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Now marker labelling
    et, markers = cv.connectedComponents(sure_fg)
 
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # Now compute watershed
    markers = cv.watershed(image, markers)
    new_masks = np.zeros(image.shape)

    new_masks[markers == -1] = [255, 0, 0]

    plt.figure()
    plt.imshow(new_masks)
    plt.show()


