# Assignment 2

## Part 1

### (a)
Write a computer program to implement a delta modulation based encoder and decoder.

### (b)
Apply it on a monochrome image from the image archive. Use the following values: (i) ς = ±5 (ii) ς = ±10(iii) ς = ±20. Observe the reconstructed image in each case and comment on the granular noise and slope overload observed, if any.

### (c)
In each case, compute the PSNR of the reconstructed image.

## Part 2

### (a)
Apply the fourth order prediction equation on a monochrome image with coef. values 0.1, 0.4, 0.1, 0.4 and obtain the predicted image. Consider all pixels beyond the boundary to have the nearest boundary pixel value.

### (b)
Obtain the error image in (a) and plot the error image histogram.

### (c)
Determine the error image variance, model its distribution as Laplacian and design Lloyd-Max quantizers for (i) 2, (ii) 4 and (iii) 8 levels, based on the Lloyd Max quantization table for unit variance Laplacian.

### (d)
Write a computer program to implement a DPCM encoder and decoder.

### (e)
For each of the quantizers designed in (c), obtain the reconstructed images and compute their PSNR values.
