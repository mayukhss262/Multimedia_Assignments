import numpy as np
from PIL import Image
import importlib

idct_module = importlib.import_module("1a_idct_8x8")
idct_8x8 = idct_module.idct_8x8

# Load DCT coefficients
dct_image = np.loadtxt("dct_coefficients.txt", dtype=np.float64)
H, W = dct_image.shape
print(f"Loaded DCT coefficient matrix: {H}x{W}")

# Apply IDCT block-by-block
reconstructed = np.zeros((H, W), dtype=np.float64)

for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = dct_image[i:i+8, j:j+8]
        reconstructed[i:i+8, j:j+8] = idct_8x8(block)

# Save reconstructed pixel values (double precision)
np.savetxt("idct_reconstructed_pixels.txt", reconstructed, fmt="%24.15e", delimiter="\t")
print(f"Reconstructed pixel matrix saved to idct_reconstructed_pixels.txt")

# Convert to uint8 and save as PNG
reconstructed_uint8 = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
img = Image.fromarray(reconstructed_uint8, mode="L")
img.save("lena_reconstructed_exact.png")
print(f"Reconstructed image saved to lena_reconstructed_exact.png")

print(f"\nBlock size: 8x8")
print(f"Number of blocks: {(H // 8) * (W // 8)}")
