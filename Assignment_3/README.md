# Assignment 3

## Contents

### Python Scripts

| Script | Description |
|--------|-------------|
| `kl_transform_foreman.py` | Performs the K-L Transform on the first 6 frames of the Foreman video sequence. Computes the mean vector, covariance matrix, eigenvectors, eigenvalues, and principal component images. Applies inverse transformation retaining only the top 2 eigenvectors and calculates the PSNR of the reconstructed images. |

### Data Files

| File/Folder | Description |
|-------------|-------------|
| `foreman_frames/` | Folder containing grayscale PNG files of the first 6 frames of the Foreman video sequence. |
| `output_images/` | Folder containing the generated principal component images and the reconstructed frames after the inverse K-L transform. |
| `results.txt` | Complete record of the calculations: mean vector, covariance matrix, eigenvalues, variance percentage, eigenvectors, and PSNR values. |
| `[Multimedia] Group 1 - Assignment 3 Report.pdf` | Complete assignment report. |

### How to Run

```bash
python kl_transform_foreman.py
```

### Dependencies

- **Python** 3.8+
- **NumPy** — matrix operations
- **Pillow** — image I/O
