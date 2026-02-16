import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Apply quantization to DCT coefficients")
parser.add_argument("--input", required=True, help="Input DCT coefficient matrix file")
parser.add_argument("--qmat", required=True, help="Path to quantization matrix file")
parser.add_argument("--output", required=True, help="Output file for quantized matrix")
args = parser.parse_args()

# Load DCT coefficients and quantization matrix
dct_image = np.loadtxt(args.input, dtype=np.float64)
Q = np.loadtxt(args.qmat, dtype=np.float64)

H, W = dct_image.shape
print(f"Loaded DCT coefficient matrix: {H}x{W}")
print(f"Quantization matrix: {args.qmat} ({Q.shape[0]}x{Q.shape[1]})")

# Apply quantization block-by-block: S(u,v) = NINT(S(u,v) / T(u,v))
quantized = np.zeros((H, W), dtype=np.float64)

for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = dct_image[i:i+8, j:j+8]
        quantized[i:i+8, j:j+8] = np.round(block / Q)

# Save quantized coefficients
np.savetxt(args.output, quantized, fmt="%8.1f", delimiter="\t")
print(f"Quantized matrix saved to {args.output}")

print(f"\nBlock size: 8x8")
print(f"Number of blocks: {(H // 8) * (W // 8)}")
print(f"Non-zero coefficients: {np.count_nonzero(quantized)} / {quantized.size}")
print(f"Zero coefficients: {quantized.size - np.count_nonzero(quantized)} / {quantized.size}")
