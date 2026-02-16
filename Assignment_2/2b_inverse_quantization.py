import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Apply inverse quantization to quantized DCT coefficients")
parser.add_argument("--qmat", required=True, help="Path to quantization matrix file")
parser.add_argument("--input", required=True, help="Input quantized matrix file")
parser.add_argument("--output", required=True, help="Output file for inverse quantized matrix")
args = parser.parse_args()

# Load quantized coefficients and quantization matrix
quantized = np.loadtxt(args.input, dtype=np.float64)
Q = np.loadtxt(args.qmat, dtype=np.float64)

H, W = quantized.shape
print(f"Loaded quantized matrix: {H}x{W}")
print(f"Quantization matrix: {args.qmat} ({Q.shape[0]}x{Q.shape[1]})")

# Apply inverse quantization block-by-block: multiply by Q
inv_quantized = np.zeros((H, W), dtype=np.float64)

for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = quantized[i:i+8, j:j+8]
        inv_quantized[i:i+8, j:j+8] = block * Q

# Save inverse quantized coefficients
np.savetxt(args.output, inv_quantized, fmt="%24.15e", delimiter="\t")
print(f"Inverse quantized matrix saved to {args.output}")

print(f"\nBlock size: 8x8")
print(f"Number of blocks: {(H // 8) * (W // 8)}")
