import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Verify if two images are identical")
parser.add_argument("--original", required=True, help="Original image file")
parser.add_argument("--reconstructed", required=True, help="Reconstructed image file")
args = parser.parse_args()

# Load images
original = np.array(Image.open(args.original).convert("L"))
reconstructed = np.array(Image.open(args.reconstructed).convert("L"))

print(f"Original shape: {original.shape}, dtype: {original.dtype}")
print(f"Reconstructed shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")

if original.shape != reconstructed.shape:
    print("\nFAILED: Images have different dimensions!")
elif np.array_equal(original, reconstructed):
    print("\nPASSED: Images are exactly identical!")
else:
    diff = np.abs(original.astype(int) - reconstructed.astype(int))
    num_diff = np.count_nonzero(diff)
    print(f"\nFAILED: Images differ at {num_diff} pixels out of {original.size}")
