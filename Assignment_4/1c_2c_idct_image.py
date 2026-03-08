import numpy as np
from PIL import Image
import importlib
import argparse

parser = argparse.ArgumentParser(description="Apply IDCT on coefficient matrix and reconstruct image")
parser.add_argument("--input", required=True, help="Input coefficient matrix file")
parser.add_argument("--output", required=True, help="Output reconstructed PNG file")
parser.add_argument("--matrix-output", required=True, help="Output IDCT matrix text file")
args = parser.parse_args()

idct_module = importlib.import_module("1a_idct_8x8")
idct_8x8 = idct_module.idct_8x8

# Load coefficients
dct_image = np.loadtxt(args.input, dtype=np.float64)
H, W = dct_image.shape
print(f"Loaded coefficient matrix: {args.input} ({H}x{W})")

# Apply IDCT block-by-block
reconstructed = np.zeros((H, W), dtype=np.float64)

for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = dct_image[i:i+8, j:j+8]
        reconstructed[i:i+8, j:j+8] = idct_8x8(block)

# Save reconstructed pixel values (double precision)
np.savetxt(args.matrix_output, reconstructed, fmt="%24.15e", delimiter="\t")
print(f"Reconstructed pixel matrix saved to {args.matrix_output}")

# Convert to uint8 and save as PNG
reconstructed_uint8 = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
img = Image.fromarray(reconstructed_uint8, mode="L")
img.save(args.output)
print(f"Reconstructed image saved to {args.output}")

print(f"\nBlock size: 8x8")
print(f"Number of blocks: {(H // 8) * (W // 8)}")
