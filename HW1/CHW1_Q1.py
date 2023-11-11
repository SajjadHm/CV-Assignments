#!/usr/bin/env python
# coding: utf-8

# # Part1

# In[39]:


# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[45]:


# This function gets a grayscale image and return equalized image
def histogram_equalization(image):
    result = np.copy(image)
    
    # number of pixels
    MN = image.shape[0]*image.shape[1]
    equ_level = np.round(np.cumsum(cv2.calcHist([image], [0], None,
                            [256],[0,256])/MN)*255).astype(np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = equ_level[image[i, j]]
            
    return result


# In[47]:


# Load images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# In[48]:


#apply Histogram Equalization on images
image1_gray_equalized = histogram_equalization(image=image1_gray)
image2_gray_equalized = histogram_equalization(image=image2_gray)


# In[83]:


# Plot images and histograms before and after equalization
# image1
plt.figure(figsize=(10, 14))
plt.subplot(1, 2, 1)
plt.imshow(image1_gray, vmin=0, vmax=255, cmap='gray')
plt.title('Image1')
plt.axis('off')

# equalized image1
plt.subplot(1, 2, 2)
plt.imshow(image1_gray_equalized, vmin=0, vmax=255, cmap='gray')
plt.title('Image1 Equalized')
plt.axis('off')
plt.show()


# In[84]:


# Plot original Histogram
plt.figure(figsize=(10, 5))
fig, ax = plt.subplots(1,2 , sharey=True)
ax[0].hist(image1_gray.ravel(), bins=256, range=[0, 256], 
           color='blue')
ax[0].set_title('Histogram of Original Image1')
ax[0].set_xlabel('Intensity')
ax[0].set_ylabel('Frequency')

# Plot equalized Histogram
ax[1].hist(image1_gray_equalized.ravel(), bins=256, range=[0, 256], 
           color='green')
ax[1].set_title('Histogram of Equalized Image1')
ax[1].set_xlabel('Intensity')
ax[1].set_ylabel('Frequency')
plt.show()


# In[85]:


# Plot images and histograms before and after equalization
# Plot image2
plt.figure(figsize=(10, 14))
plt.subplot(1, 2, 1)
plt.imshow(image2_gray, vmin=0, vmax=255, cmap='gray')
plt.title('Image2')
plt.axis('off')

# Plot equalized image2
plt.subplot(1, 2, 2)
plt.imshow(image2_gray_equalized, vmin=0, vmax=255, cmap='gray')
plt.title('Image2 Equalized')
plt.axis('off')
plt.show()


# In[86]:


# Original Histogram
plt.figure(figsize=(10, 5))
fig, ax = plt.subplots(1,2 , sharey=True)
ax[0].hist(image2_gray.ravel(), bins=256, range=[0, 256], 
           color='blue')
ax[0].set_title('Histogram of Original Image2')
ax[0].set_xlabel('Intensity')
ax[0].set_ylabel('Frequency')

# Equalized Histogram
ax[1].hist(image2_gray_equalized.ravel(), bins=256, range=[0, 256], 
           color='green')
ax[1].set_title('Histogram of Equalized Image2')
ax[1].set_xlabel('Intensity')
ax[1].set_ylabel('Frequency')
plt.show()


# # Part 2

# - A)

# در تصویر اول نواحی که میزان روشنایی زیاد است در تصویر خروجی به مراتب جزییات بیشتری مشخص شده و به نحوی میتواند در بازیابی قسمت های از دست رفته به ما کمک کند.نواحی مانند پشت سر هیتلر.

# - B)

#  .در تصویر دوم نواحی  بعد از هموار سازی هیستوگرام مانند بالای لب تفاوت مشهودی میان دو شخصیت مشاهده میشود .علاوه براین با هموار سازی هستوگرام در این تصویر چهره ها روشن تر شده و جزییات چهره ها واضح تر شده است که میتواند در تشخیص اثر کپی به ما کمک کند.
