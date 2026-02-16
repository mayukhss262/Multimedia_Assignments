# Assignment 2

## Part 1

### (a)
Write a computer program to perform DCT and IDCT over an 8×8 array.

### (b)
Subdivide a monochrome image into non-overlapping blocks of size 8×8 and apply DCT on each block to obtain the transform coefficients.

### (c)
Apply IDCT on each block of transformed coefficients and obtain the reconstructed image.

### (d)
Check that the reconstructed image is exactly the same as that of the original image.

---

**Note:**  
Represent the DCT kernel and the transformed coefficient values in double-precision floating point and do not truncate your results, before converting the reconstructed image into an unsigned character array.

## Part 2

### (a)
On the 8×8 transformed coefficients obtained in the above assignment, apply the following quantization matrix and obtain the quantized DCT coefficients.

Quantization Matrix:

| 16 | 11 | 10 | 16 | 24 | 40 | 51 | 61 |
| 12 | 12 | 14 | 19 | 26 | 58 | 60 | 55 |
| 14 | 13 | 16 | 24 | 40 | 57 | 69 | 56 |
| 14 | 17 | 22 | 29 | 51 | 87 | 80 | 62 |
| 18 | 22 | 37 | 56 | 68 | 109 | 103 | 77 |
| 24 | 35 | 55 | 64 | 81 | 104 | 113 | 92 |
| 49 | 64 | 78 | 87 | 103 | 121 | 120 | 101 |
| 72 | 92 | 95 | 98 | 112 | 100 | 103 | 99 |

### (b)
Obtain inverse quantization by multiplying the quantized DCT coefficients with the corresponding elements of the quantization matrix.

### (c)
Apply IDCT on the coefficients as above and obtain the reconstructed image.

### (d)
Check that the reconstructed image is not the same as the original image and calculate the PSNR of the reconstructed image.

### (e)
In part-(a), multiply the elements of the quantization matrix by:

- (i) 2  
- (ii) 4  
- (iii) 8  

In each case, repeat part-(a) to part-(d) and compute the PSNR of the reconstructed image.

### (f)
Check that the PSNR decreases as the multiplication factor increases.

