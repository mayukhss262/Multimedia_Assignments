# Assignments for Multimedia Systems & Applications (EC60104) [SP 25-26]

Group 1 
Members : 
Mayukh Shubhra Saha - 21EC37009
Adipta Halder - 21EC37023
Abhradeep De - 21EC37022

## Contents

### Python Scripts

| File | Description |
|------|-------------|
| `1a_generate_bit_plane_images.py` | Extracts 8 bit planes (bit 0–7) from a grayscale image and saves them as binary images. |
| `1b_generate_graycode_images.py` | Converts image to Gray code representation, then extracts 8 bit planes. |
| `2a_run_length_coding.py` | Applies 6-bit fixed-length RLE to both binary and Gray-coded bit planes; compares compression ratios. |
| `2b_huffman_common.py` | Performs Huffman coding on RLE symbols using a single codebook shared across all bit planes. |
| `2b_huffman_individual.py` | Performs Huffman coding on RLE symbols using individual codebooks per bit plane. |
| `predictive_coding_q3.py` | Implements predictive coding techniques for image compression. |

### Output Directories

| Directory | Description |
|-----------|-------------|
| `bit_plane_images/` | Contains 8 binary bit-plane images extracted from the source image (from `1a`). |
| `gray_coded_bit_plane_images/` | Contains Gray-coded image and 8 bit-plane images (from `1b`). |
| `RL_bit_plane_images/` | RLE-compressed binary files and compression stats for binary bit planes (from `2a`). |
| `RL_gray_coded_bit_plane_images/` | RLE-compressed binary files and compression stats for Gray-coded bit planes (from `2a`). |

### Result Files

| File | Description |
|------|-------------|
| `2b_huffman_common_results.txt` | Detailed Huffman coding results with common codebook. Complete compression statistics across all bit planes and the complete Huffman codebook and symbol statistics are included. |
| `2b_huffman_individual_results.txt` | Detailed Huffman coding results with individual codebooks per plane. Complete compression statistics across all bit planes and the complete Huffman codebooks and symbol statistics are included. |
| `1c_comparisons.txt` | Comparison analysis of binary vs Gray-coded bit planes. |
| `DOCUMENTATION_Q3.md` | Documentation for predictive coding (Question 3). |

### Input

| File | Description |
|------|-------------|
| `lena.png` | Source grayscale test image (225×225). |

## Usage

Run the scripts in order. Each script depends on outputs from previous scripts.

```bash
# Step 1a: Generate binary bit-plane images
python 1a_generate_bit_plane_images.py

# Step 1b: Generate Gray-coded bit-plane images
python 1b_generate_graycode_images.py

# Step 2a: Apply Run-Length Encoding and compare compression
python 2a_run_length_coding.py

# Step 2b: Apply Huffman coding (common codebook)
python 2b_huffman_common.py

# Step 2b: Apply Huffman coding (individual codebooks)
python 2b_huffman_individual.py

# Question 3: Predictive coding
python predictive_coding_q3.py
```

## Requirements

- Python 3.x
- NumPy
- Pillow (PIL)
