import os
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error


def rmse(true_images, pred_images):
    result, n = 0, 0
    for true_img, pred_img in zip(true_images, pred_images):
        result += np.sum((true_img.ravel()/255.0 - pred_img.ravel()/255.0)**2)
        n += len(true_img.ravel())
    return (result / float(n))**0.5


def clean_image(input_img):
    kernel = np.ones((4,4), np.uint8) 
    #erode will remove only background
    img_erode  = 255- cv2.erode(255-input_img, kernel,iterations = 1)
    img_sub = cv2.add(input_img, - img_erode)
    #need to choose threshold automatically?
    _, img_thresh = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)
    mask = img_thresh == 0                                     
    img_final = np.where(mask, input_img, 255)
    return img_final



train_X_images = [cv2.imread(os.path.join('train', fname), cv2.IMREAD_GRAYSCALE) 
                                                      for fname in os.listdir('train') ]
train_y_images = [cv2.imread(os.path.join('train_cleaned', fname), cv2.IMREAD_GRAYSCALE)
                                                       for fname in os.listdir('train_cleaned')]

print "train data RMSE without cleaning: "
print rmse(train_y_images, train_X_images)


print "train data RMSE with cleaning: " 
print  rmse(train_y_images, map(clean_image, train_X_images))



#apply to test set
test_X_images = [cv2.imread(os.path.join('test', fname), cv2.IMREAD_GRAYSCALE) 
                                                       for fname in os.listdir('test') ]
predicted_test_images = map(clean_image, test_X_images)
for fname, img in zip(os.listdir('test'), predicted_test_images):
    cv2.imwrite(os.path.join('test_cleaned', fname), img)
