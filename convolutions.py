from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
	#Grab the spatial dimensions of the image and kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]

	#Allocate memory for the output image, taking care to "pad"
	#the borders of the input image, so the spartial size (i. e.,
	#width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, 
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float")

	#loop over the input image, sliding the kernel across
	#each (x, y) coordinates from left to right and top to bottom

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			#Extract the ROI of the image by extracting the
			#center region of the current (x, y) coodinates
			#dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			#perform the actual convolution by taking the 
			#element wise multiplication between the ROI and
			#the kernel, then summing the matrix
			k = (roi * K).sum()

			#store the convolved value in the output (x, y)
			#coordinate of the output image
			output[y - pad, x - pad] = k

	#rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	#return the output image
	return output

#Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

#Construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

#Construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

#Construct the Laplacian kernel used to detect edge like
#regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

#Construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

#Construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

#Construct the emboss
emboss = np.array((
	[-2, -1, 0],
	[-1, 1, 1],
	[0, 1, 2]), dtype="int")

'''Construct the kernel bank, a list of kernels we're going
to apply using both our custom convolve function and
OpenCV filter2d function'''
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY),
	("emboss", emboss))

#Load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Loop over the kernels
for (kernelName, K) in kernelBank:
	#Apply the kernel to the grayscale image using both our
	#custom convolve function and OpenCV filter2d function
	print("[INFO] applying {} kernel".format(kernelName))
	convolveOutput = convolve(gray, K)
	opencvOutput = cv2.filter2D(gray, -1, K)

	#Show the output images
	cv2.imshow("Original", gray)
	cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()