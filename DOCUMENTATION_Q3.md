# LOSSLESS PREDICTIVE CODING - IMPLEMENTATION DOCUMENTATION

## 1. Algorithm Overview

### 1.1 High-Level Concept
Lossless predictive coding exploits **spatial correlation** in natural images. Adjacent pixels in natural images tend to have similar values, so we can predict a pixel's value from its neighbors. The prediction error (residual) has much lower entropy than the original image, enabling efficient compression.

### 1.2 Mathematical Foundation
The current pixel $s(n_1, n_2)$ is predicted using a linear combination of **four causal neighbors**:

$$\hat{s}(n_1, n_2) = a_1 \cdot s(n_1-1, n_2-1) + a_2 \cdot s(n_1-1, n_2) + a_3 \cdot s(n_1-1, n_2+1) + a_4 \cdot s(n_1, n_2-1)$$

Where:
- $a_1, a_2, a_3, a_4$ are prediction coefficients (all 0.25 in our implementation)
- Constraint: $a_1 + a_2 + a_3 + a_4 = 1$

**Spatial Neighbor Layout:**
```
     s(n₁-1, n₂-1)   s(n₁-1, n₂)   s(n₁-1, n₂+1)
                     s(n₁, n₂-1)   [CURRENT PIXEL]
```

**Special Boundary Handling:**
For the **last column** ($n_2 = W-1$), the top-right neighbor $s(n_1-1, n_2+1)$ is outside the image boundaries. In this specific case, we redistribute the weight of $a_3$ proportionally among the other three available neighbors to maintain the sum of coefficients equals to 1.

---

## 2. Implementation Details

### 2.1 Prediction Formula
```python
predicted[n1, n2] = (
    0.25 * s[n1 - 1, n2 - 1] +  # Top-left diagonal
    0.25 * s[n1 - 1, n2] +       # Top
    0.25 * s[n1 - 1, n2 + 1] +   # Top-right diagonal
    0.25 * s[n1, n2 - 1]         # Left
)
```

### 2.2 Boundary Condition Handling

| Position | Available Neighbors | Handling Strategy |
|----------|---------------------|-------------------|
| `(0, 0)` - First pixel | None | Predict as 0 |
| First row (`n₁ = 0`) | Only left `s(0, n₂-1)` | Use left neighbor as sole predictor |
| First column (`n₂ = 0`) | Only top `s(n₁-1, 0)` | Use top neighbor as sole predictor |
| Last column (`n₂ = W-1`) | No top-right diagonal | Redistribute weights among available neighbors |
| Interior pixels | All 4 neighbors | Apply full prediction formula |

**Last Column Handling (Code):**
```python
elif n2 == W - 1:
    total_avail = a1 + a2 + a4  # Exclude a3
    w1, w2, w4 = a1/total_avail, a2/total_avail, a4/total_avail
    predicted[n1, n2] = w1*s[n1-1,n2-1] + w2*s[n1-1,n2] + w4*s[n1,n2-1]
```

### 2.3 Numerical Considerations

1. **Rounding**: Predicted values are rounded to integers using `np.round()` for lossless reconstruction
2. **Data Types**: 
   - Original image: `int16` (to handle arithmetic)
   - Error image: `int16` (range: [-255, +255])
3. **Display Offset**: Error image displayed with +127 offset to visualize in grayscale range

---

## 3. Data Structures

| Array | Type | Shape | Range | Purpose |
|-------|------|-------|-------|---------|
| `original_image` | `np.int16` | `(H, W)` | [0, 255] | Input grayscale image |
| `predicted_image` | `np.int16` | `(H, W)` | [0, 255] | Predicted pixel values |
| `error_image` | `np.int16` | `(H, W)` | [-255, 255] | Prediction residuals |

---

## 4. Function Specifications

### 4.1 `__init__(self, image_input, coefficients)`
- **Purpose**: Initialize coder with image and coefficients
- **Inputs**: Path string or numpy array, coefficient tuple
- **Validation**: Coefficients must sum to 1.0, image must be grayscale

### 4.2 `compute_predicted_image(self)`
- **Purpose**: Generate predicted image using 4-neighbor model
- **Algorithm**: Iterate through all pixels, apply appropriate prediction based on position
- **Complexity**: O(H × W) time, O(H × W) space
- **Returns**: `np.ndarray` of predicted values

### 4.3 `compute_error_image(self)`
- **Purpose**: Compute residuals: `error = original - predicted`
- **Complexity**: O(H × W)
- **Returns**: `np.ndarray` of signed error values

### 4.4 `compute_entropy(self, image)`
- **Purpose**: Calculate Shannon entropy
- **Formula**: $H = -\sum_i P(i) \cdot \log_2(P(i))$
- **Returns**: Entropy in bits per pixel

### 4.5 `verify_lossless_reconstruction(self)`
- **Purpose**: Verify `original == predicted + error`
- **Returns**: Tuple `(is_lossless: bool, max_error: float)`

---

## 5. Usage Guide

### 5.1 Command Line
```bash
# Using a specific image
python predictive_coding_q3.py --image your_image.png

# Using default synthetic image
python predictive_coding_q3.py

# Validation with 4x4 test image
python predictive_coding_q3.py --test

# Custom output directory
python predictive_coding_q3.py --image img.png --output results/
```

### 5.2 Programmatic Usage
```python
from predictive_coding_q3 import LosslessPredictiveCoder

coder = LosslessPredictiveCoder('image.png')
coder.compute_predicted_image()
coder.compute_error_image()

# Get entropy analysis
H_orig = coder.compute_entropy(coder.original_image)
H_err = coder.compute_entropy(coder.error_image)
print(f"Entropy reduction: {100*(H_orig-H_err)/H_orig:.1f}%")

# Verify lossless property
is_lossless, max_err = coder.verify_lossless_reconstruction()
assert is_lossless, "Reconstruction failed!"

# Save all outputs
coder.save_outputs('outputs/')
```

---

## 6. Expected Output

### 6.1 Console Output
```
=== LOSSLESS PREDICTIVE CODING ANALYSIS ===
Image: image.png
Dimensions: 512 x 512
Coefficients: a1=0.25, a2=0.25, a3=0.25, a4=0.25

--- ENTROPY ANALYSIS ---
Original Image Entropy:  7.4523 bits/pixel
Error Image Entropy:     4.2187 bits/pixel
Entropy Reduction:       43.38%

--- VERIFICATION ---
Reconstruction Test: PASS
Max reconstruction error: 0.0
```

### 6.2 Generated Files
| File | Content |
|------|---------|
| `outputs/predicted_image.png` | Predicted grayscale image |
| `outputs/error_image_display.png` | Error image with 127 offset |
| `outputs/analysis_plots.png` | 2×3 visualization grid |
| `outputs/statistics_report.txt` | Complete numerical report |

---

## 7. Testing and Validation

### 7.1 Lossless Verification Test
```python
coder = LosslessPredictiveCoder('image.png')
coder.compute_predicted_image()
coder.compute_error_image()

# This MUST pass for correct implementation
is_lossless, _ = coder.verify_lossless_reconstruction()
assert is_lossless
```

### 7.2 4×4 Test Case
```python
test_img = np.array([
    [100, 105, 110, 108],
    [102, 107, 112, 110],
    [101, 106, 111, 109],
    [103, 108, 113, 111]
], dtype=np.uint8)

coder = LosslessPredictiveCoder(test_img)
# Expect: small errors in range [-5, +5], visible entropy reduction
```

---

## 8. References

1. Lecture 5: Lossless Image Compression - Other Techniques (IIT KGP Multimedia Systems)
2. Shannon, C. E. (1948). "A Mathematical Theory of Communication"
3. Gonzalez & Woods. "Digital Image Processing" - Chapter on Lossless Compression
