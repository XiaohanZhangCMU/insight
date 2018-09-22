
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# 1 - remove the vertical line on the left

img = cv2.imread('someimage2.JPG', 0)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 100, 150, apertureSize=5)

lines = cv2.HoughLines(edges, 1, np.pi / 50, 50)
# for rho, theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))

#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)

# cv2.imshow('marked', img)
# cv2.waitKey(0)
cv2.imwrite('image.png', img)


# 2 - remove horizontal lines

img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_orig = cv2.imread("image.png")

img = cv2.bitwise_not(img)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
# cv2.imshow("th2", th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

horizontal = th2
rows, cols = horizontal.shape


# inverse the image, so that lines are black for masking
horizontal_inv = cv2.bitwise_not(horizontal)
# perform bitwise_and to mask the lines with provided mask
masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
# reverse the image back to normal
masked_img_inv = cv2.bitwise_not(masked_img)
# cv2.imshow("masked img", masked_img_inv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

horizontalsize = int(cols / 30)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
# cv2.imshow("horizontal", horizontal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# step1
edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# step2
kernel = np.ones((1, 2), dtype="uint8")
dilated = cv2.dilate(edges, kernel)
# cv2.imshow("dilated", dilated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

im2, ctrs, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y + h, x:x + w]

    # show ROI
    rect = cv2.rectangle(img_orig, (x, y), (x + w, y + h), (255, 255, 255), -1)

# cv2.imshow('areas', rect)
# cv2.waitKey(0)

cv2.imwrite('no_lines.png', rect)


# 3 - detect and extract ROI's

image = cv2.imread('no_lines.png')
# cv2.imshow('i', image)
# cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

# dilation
kernel = np.ones((8, 45), np.uint8)  # values set for this image only - need to change for different images
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilated', img_dilation)
# cv2.waitKey(0)

# find contours
im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # cv2.waitKey(0)

    # save only the ROI's which contain a valid information
#    if h > 10 and w > 75:
    cv2.imwrite('roi_{}.png'.format(i), roi)

    bw_image = cv2.imread('roi_{}.png'.format(i))
    bw_gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
    bw_ret, bw_thresh = cv2.threshold(bw_gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('bw_roi_{}.png'.format(i), bw_thresh)
# cv2.imshow('marked areas', image)
# cv2.waitKey(0)


# In[45]:


bw_image = cv2.imread('roi_{}.png'.format(i))
bw_gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
bw_ret, bw_thresh = cv2.threshold(bw_gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('bw_roi_{}.png'.format(i), bw_thresh)


# In[47]:


# tmp = cv2.imread('bw_roi_{}.png'.format(i))


# In[56]:


# np.min(tmp[:,:,2])

