# Assignment 2

## Contents

### Python Scripts

| Script | Description |
|--------|-------------|
| `delta_modulation.py` | Implements Delta Modulation (DM) on a monochrome image with step sizes ζ = 5, 10, 20. Saves reconstructed images and computes PSNR for each. |
| `dpcm.py` | Implements Differential Pulse Code Modulation (DPCM) with open-loop prediction and Lloyd-Max quantization (2, 4, 8 levels). Saves error image, error histogram, reconstructed images, and computes PSNR. |

### Data Files

| File | Description |
|------|-------------|
| `lena.png` | Original 256×256 monochrome test image. |
| `part_1_dm_results.txt` | Delta Modulation results: observations on slope overload/granular noise and PSNR values. |
| `part_2_dpcm_results.txt` | DPCM results: error variance, quantizer thresholds, reconstruction levels, and PSNR values. |
| `reconstructed_zeta_5.png` | DM reconstructed image with ζ = 5 (PSNR: 17.08 dB). |
| `reconstructed_zeta_10.png` | DM reconstructed image with ζ = 10 (PSNR: 19.54 dB). |
| `reconstructed_zeta_20.png` | DM reconstructed image with ζ = 20 (PSNR: 21.48 dB). |
| `dpcm_error_image.png` | DPCM open-loop prediction error image (scaled for visibility). |
| `dpcm_error_histogram.png` | Histogram of DPCM prediction errors. |
| `dpcm_reconstructed_2_levels.png` | DPCM reconstructed image with 2-level quantizer (PSNR: 22.17 dB). |
| `dpcm_reconstructed_4_levels.png` | DPCM reconstructed image with 4-level quantizer (PSNR: 27.27 dB). |
| `dpcm_reconstructed_8_levels.png` | DPCM reconstructed image with 8-level quantizer (PSNR: 31.67 dB). |
| `[Multimedia] Group 1 - Assignment 2 Report.pdf` | Complete assignment report. |

### How to Run

```bash
# Part 1: Delta Modulation
python delta_modulation.py

# Part 2: DPCM
python dpcm.py
```

### Dependencies

- **Python** 3.8+
- **NumPy** — matrix operations
- **Pillow** — image I/O (`PIL.Image`)
- **Matplotlib** — histogram plotting


