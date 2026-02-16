# Assignment 2 — DCT/IDCT Image Compression & Quantization

## Python Scripts

| Script | Description |
|--------|-------------|
| `1a_dct_8x8.py` | Defines `dct_8x8()` — performs 2D DCT on an 8×8 block using scipy. |
| `1a_idct_8x8.py` | Defines `idct_8x8()` — performs 2D IDCT on an 8×8 block using scipy. |
| `1b_dct_image.py` | Loads a monochrome image, divides into 8×8 blocks, applies DCT, saves coefficient matrix and pixel matrix. |
| `1c_2c_idct_image.py` | Loads a coefficient matrix, applies IDCT block-by-block, saves reconstructed pixel matrix and PNG image. |
| `1d_verify_reconstruction.py` | Compares two PNG images pixel-by-pixel and reports if they are identical. |
| `2a_quantization.py` | Quantizes DCT coefficients by dividing each 8×8 block element-wise by a quantization matrix and rounding. |
| `2b_inverse_quantization.py` | Performs inverse quantization by multiplying quantized coefficients with the quantization matrix element-wise. |
| `2d_psnr.py` | Checks if two images are identical, then computes PSNR of reconstructed image vs original. |

## Data Files

| File | Description |
|------|-------------|
| `lena.png` | Original 256×256 monochrome test image. |
| `quantization_matrix.txt` | Standard JPEG 8×8 luminance quantization matrix. |
| `quantization_matrix_{2,4,8}.txt` | Scaled quantization matrices (elements multiplied by 2, 4, 8). |
| `original_pixels.txt` | Original image pixel values as a 256×256 matrix. |
| `dct_coefficients.txt` | 256×256 DCT coefficient matrix (double precision). |
| `quantized_matrix{,_2,_4,_8}.txt` | Quantized DCT coefficients for each quantization scale. |
| `inv_quantized_matrix{,_2,_4,_8}.txt` | Inverse-quantized (dequantized) DCT coefficients. |
| `idct_reconstructed_pixels.txt` | IDCT output from exact (unquantized) DCT coefficients. |
| `idct_dequantized_pixels{,_2,_4,_8}.txt` | IDCT output from dequantized coefficients at each scale. |
| `lena_reconstructed_exact.png` | Reconstructed image from exact DCT→IDCT (should match original). |
| `lena_reconstructed_quantized{,_2,_4,_8}.png` | Reconstructed images after quantization at each scale. |

## How to Run

```bash
# Part 1: DCT, IDCT, and verification
python 1b_dct_image.py --input lena.png
python 1c_2c_idct_image.py --input dct_coefficients.txt --output lena_reconstructed_exact.png --matrix-output idct_reconstructed_pixels.txt
python 1d_verify_reconstruction.py --original lena.png --reconstructed lena_reconstructed_exact.png

# Part 2: Quantization, inverse quantization, reconstruction, and PSNR
python 2a_quantization.py --input dct_coefficients.txt --qmat quantization_matrix.txt --output quantized_matrix.txt
python 2b_inverse_quantization.py --qmat quantization_matrix.txt --input quantized_matrix.txt --output inv_quantized_matrix.txt
python 1c_2c_idct_image.py --input inv_quantized_matrix.txt --output lena_reconstructed_quantized.png --matrix-output idct_dequantized_pixels.txt
python 2d_psnr.py --original lena.png --reconstructed lena_reconstructed_quantized.png

# Repeat Part 2 with scaled quantization matrices (_2, _4, _8) by substituting filenames accordingly.
```

## Dependencies

- **Python** 3.8+
- **NumPy** — matrix operations
- **SciPy** — `scipy.fft.dct` / `scipy.fft.idct`
- **Pillow** — image I/O (`PIL.Image`)

Install: `pip install numpy scipy pillow`
