# Assignment 3

### (a)
Consider the first six frames of the video sequence “Foreman”. Compose 6-element vectors by picking up pixel values at the same spatial position over six consecutive frames.

### (b)
Determine the mean of these 6-element vectors, considering all spatial positions. Compute the 6x6 covariance matrix and determine its eigenvalues and the corresponding eigenvectors. Retain only top two of these eigenvalues and the corresponding eigenvectors.

### (c)
Obtain the top two principal component images by projecting the vectors (obtaining dot-products) on the two principal eigenvectors and display the results. 

### (d)
Apply inverse K-L transformation and obtain the reconstructed frames. Compute the PSNR of each reconstructed frame.