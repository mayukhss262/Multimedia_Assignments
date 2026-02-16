import numpy as np
from PIL import Image

# Load images
original = np.array(Image.open("lena.png").convert("L"))
reconstructed = np.array(Image.open("lena_reconstructed_exact.png").convert("L"))

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
