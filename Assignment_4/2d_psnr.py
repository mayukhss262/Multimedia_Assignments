import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Verify reconstruction and compute PSNR")
parser.add_argument("--original", required=True, help="Original image file")
parser.add_argument("--reconstructed", required=True, help="Reconstructed image file")
args = parser.parse_args()

# Load images
original = np.array(Image.open(args.original).convert("L"), dtype=np.float64)
reconstructed = np.array(Image.open(args.reconstructed).convert("L"), dtype=np.float64)

print(f"Original: {args.original} ({original.shape[0]}x{original.shape[1]})")
print(f"Reconstructed: {args.reconstructed} ({reconstructed.shape[0]}x{reconstructed.shape[1]})")

# Check if identical
if np.array_equal(original.astype(np.uint8), reconstructed.astype(np.uint8)):
    print("\nImages are exactly identical!")
    print("PSNR: Infinity (no difference)")
else:
    diff = original - reconstructed
    M, N = original.shape
    sum_sq_diff = np.sum(diff ** 2)

    # PSNR = 20 * log10( sqrt(255^2 * M * N / sum_sq_diff) )
    psnr = 20 * np.log10(np.sqrt((255 ** 2) * M * N / sum_sq_diff))

    print(f"\nImages are NOT identical.")
    print(f"PSNR: {psnr:.4f} dB")
