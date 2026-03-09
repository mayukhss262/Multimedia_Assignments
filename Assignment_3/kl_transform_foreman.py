import numpy as np
from PIL import Image
import os

SCRIPT_DIR = os.getcwd()
FRAME_DIR = os.path.join(SCRIPT_DIR, "foreman_frames")
NUM_FRAMES = 6
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "results.txt")
OUTPUT_IMG_DIR = os.path.join(SCRIPT_DIR, "output_images")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

frames = []
for i in range(1, NUM_FRAMES + 1):
    path = os.path.join(FRAME_DIR, f"foreman_frame_{i:03d}.png")
    img = np.array(Image.open(path), dtype=np.float64)
    frames.append(img)

H, W = frames[0].shape
N = H * W  # total number of spatial positions

X = np.stack([f.flatten() for f in frames], axis=0)  # (6, N)

mean_vector = np.mean(X, axis=1)  # (6,)

X_centered = X - mean_vector[:, np.newaxis]  # (6, N)

cov_matrix = (X_centered @ X_centered.T) / N  # (6, 6)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort in descending order of eigenvalues
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

# Top 2 eigenvalues
top2_eigenvalues = eigenvalues[:2]
top2_eigenvectors = eigenvectors[:, :2]

with open(OUTPUT_FILE, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Assignment 3 Results\n")
    f.write(f"Frame size: {W} x {H} pixels (grayscale)\n")
    f.write(f"Total number of vectors: {N}\n")
    f.write("=" * 60 + "\n\n")

    f.write("MEAN VECTOR:\n")
    f.write("-" * 40 + "\n")
    for val in mean_vector:
        f.write(f"  {val:.6f}\n")
    f.write("\n")

    f.write("COVARIANCE MATRIX:\n")
    f.write("-" * 40 + "\n")
    for i in range(NUM_FRAMES):
        f.write("  [" + ", ".join(f"{cov_matrix[i, j]:12.4f}" for j in range(NUM_FRAMES)) + "]\n")
    f.write("\n")

    f.write("EIGENVALUES AND CORRESPONDING EIGENVECTORS:\n")
    f.write("-" * 105 + "\n")
    f.write(f"  {'Eigenvalue':<18s}  {'Eigenvector':<72s}  {'Status'}\n")
    f.write("-" * 105 + "\n")
    for i in range(NUM_FRAMES):
        marker = "RETAINED" if i < 2 else "DISCARDED"
        ev_str = "[" + ", ".join(f"{eigenvectors[k, i]:10.6f}" for k in range(NUM_FRAMES)) + "]"
        f.write(f"  {eigenvalues[i]:<18.6f}  {ev_str:<72s}  {marker}\n")
    f.write("\n")

PC = top2_eigenvectors.T @ X_centered  # (2, N)

for k in range(2):
    pc_flat = PC[k, :]  # (N,)
    # Normalize to [0, 255] for display
    pc_min = pc_flat.min()
    pc_max = pc_flat.max()
    pc_normalized = ((pc_flat - pc_min) / (pc_max - pc_min) * 255).astype(np.uint8)
    pc_image = pc_normalized.reshape((H, W))
    
    output_path = os.path.join(OUTPUT_IMG_DIR, f"principal_component_{k+1}.png")
    Image.fromarray(pc_image).save(output_path)


X_reconstructed = top2_eigenvectors @ PC + mean_vector[:, np.newaxis]  # (6, N)

psnr_values = []

for i in range(NUM_FRAMES):
    original = frames[i] 
    
    recon_flat = X_reconstructed[i, :] 
    recon_frame = np.clip(recon_flat, 0, 255).reshape((H, W))
    
    recon_uint8 = recon_frame.astype(np.uint8)
    output_path = os.path.join(OUTPUT_IMG_DIR, f"reconstructed_frame_{i+1:03d}.png")
    Image.fromarray(recon_uint8).save(output_path)
    
    mse = np.mean((original - recon_frame) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)
    
    psnr_values.append(psnr)
    
with open(OUTPUT_FILE, "a") as f:
    f.write("PSNR OF RECONSTRUCTED FRAMES:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  {'Frame':<10s}  {'PSNR (dB)':<15s}\n")
    f.write("-" * 40 + "\n")
    for i in range(NUM_FRAMES):
        f.write(f"  Frame {i+1:<4d}  {psnr_values[i]:<15.4f}\n")
    f.write("\n")

print(f"\nAll results saved to: {OUTPUT_FILE}")
print(f"Output images saved to: {OUTPUT_IMG_DIR}")
