import numpy as np
from PIL import Image
import importlib

dct_module = importlib.import_module("1a_dct_8x8")
dct_8x8 = dct_module.dct_8x8

# Load image
img = Image.open("lena.png")

# Convert to monochrome if not already
if img.mode != "L":
    img = img.convert("L")

pixels = np.array(img, dtype=np.float64)
H, W = pixels.shape

# Ensure dimensions are multiples of 8
assert H % 8 == 0 and W % 8 == 0, f"Image dimensions {H}x{W} must be multiples of 8"

# Save original pixel values
np.savetxt("original_pixels.txt", pixels, fmt="%8.1f", delimiter="\t")
print(f"Original pixel matrix saved to original_pixels.txt")

# Apply DCT block-by-block
dct_image = np.zeros((H, W), dtype=np.float64)

for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = pixels[i:i+8, j:j+8]
        dct_image[i:i+8, j:j+8] = dct_8x8(block)

# Save DCT coefficients
np.savetxt("dct_coefficients.txt", dct_image, fmt="%24.15e", delimiter="\t")
print(f"DCT coefficient matrix saved to dct_coefficients.txt")

print(f"\nInput image: lena.png ({H}x{W}, monochrome)")
print(f"Block size: 8x8")
print(f"Number of blocks: {(H // 8) * (W // 8)}")
