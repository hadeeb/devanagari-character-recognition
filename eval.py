import argparse
import numpy as np
import cv2
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

model = load_model("model.save")

image = cv2.imread(args["image"])
def predict(img):
    # image_data = (ndimage.imread(img).astype(float) - pixel_depth / 2) / pixel_depth
    image_data = img
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 1, 32, 32)).astype(np.float32)
    print(dataset.shape)
    a = model.predict(dataset)

    temp = np.sort(a, kind='mergesort')
    # print(a)
    itemindex = np.where(a==temp[0][-1])
    print("#########***#########")
    print("Imagefile = ", img)
    print("Character = ", itemindex[1][0])
    print("qwerty = ", temp[0][-1]*100,"%")
    #grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV )
cv2.imshow('second', im_bw)
cv2.waitKey(0)

#dilation
kernel = np.ones((10,10), np.uint8)
img_dilation = cv2.dilate(im_bw, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    im = img_dilation[y:y+h, x:x+w]
    desired_size = 32
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    cv2.imshow('segment no:'+str(i), new_im)
    cv2.rectangle(img_dilation, (x, y), (x + w, y + h), (90, 0, 255), 2)
    cv2.waitKey(0)
    predict(new_im)

cv2.imshow('marked areas',img_dilation)
cv2.waitKey(0)
