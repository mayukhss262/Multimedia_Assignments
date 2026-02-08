"""
Lossless Predictive Coding for Image Compression using 4-neighbor linear prediction.
The predictor uses weighted sum of neighboring pixels (top-left diagonal, top, top-right
diagonal, and left) to estimate current pixel value, and the error (residual) image has significantly lower entropy than the original.

ŝ(n₁, n₂) = a₁·s(n₁-1,n₂-1) + a₂·s(n₁-1,n₂) + a₃·s(n₁-1,n₂+1) + a₄·s(n₁,n₂-1) where coefficients a₁ = a₂ = a₃ = a₄ = 0.25 (sum to 1)

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os
from typing import Tuple, Dict, Optional, Union
import warnings


class LosslessPredictiveCoder:
    """
    Attributes:
        coefficients (tuple): Prediction coefficients (a₁, a₂, a₃, a₄)
        original_image (np.ndarray): Input grayscale image
        predicted_image (np.ndarray): Computed prediction
        error_image (np.ndarray): Prediction error (signed integers)
        image_path (str): Path to the loaded image"""
    
    def __init__(self, image_input: Union[str, np.ndarray], coefficients: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)) -> None:
        
        """Initialize the coder with an image and prediction coefficients.
        Args:
            image_input: Either path to grayscale image file or numpy array 
            coefficients: Tuple of 4 prediction coefficients (a₁, a₂, a₃, a₄)       
        Raises:
            ValueError: If coefficients don't sum to 1 or image is not grayscale
            FileNotFoundError: If image file doesn't exist
        """
        # Validate coefficients - they should sum to 1 for proper prediction
        if abs(sum(coefficients) - 1.0) > 1e-6:
            raise ValueError(f"Coefficients must sum to 1.0, got {sum(coefficients)}")
        
        self.coefficients = coefficients
        
        # Load image from file or use provided array
        if isinstance(image_input, str):
            self.image_path = image_input
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            
            # Load and convert to grayscale
            img = Image.open(image_input)
            if img.mode != 'L':
                img = img.convert('L')
                warnings.warn(f"Image converted from {img.mode} to grayscale (L)")
            
            self.original_image = np.array(img, dtype=np.int16)
        
        else:
            # Direct numpy array input
            self.image_path = "numpy_array_input"
            self.original_image = np.array(image_input, dtype=np.int16)
        
        # Validate image dimensions
        if len(self.original_image.shape) != 2:
            raise ValueError("Image must be 2D grayscale, got shape: " + str(self.original_image.shape))
        
        # Initialize output arrays
        self.predicted_image: Optional[np.ndarray] = None
        self.error_image: Optional[np.ndarray] = None
        
        # Store dimensions for convenience
        self.height, self.width = self.original_image.shape
    
    def compute_predicted_image(self) -> np.ndarray:
        """
        For each pixel at position (n₁, n₂), compute prediction using 4 causal neighbors:
            ŝ(n₁, n₂) = a₁·s(n₁-1,n₂-1) + a₂·s(n₁-1,n₂) + a₃·s(n₁-1,n₂+1) + a₄·s(n₁,n₂-1)
        
        Boundary Handling:
        - Pixel (0,0): ŝ = 0
        - First row (n₁=0): Use left neighbor
        - First column (n₂=0): Use top neighbor
        - Last column (n₂=W-1): Use top-left, top, left (redistribute a3)
        
        Returns:
            np.ndarray: Predicted image as integers
        """
        a1, a2, a3, a4 = self.coefficients
        H, W = self.height, self.width
        s = self.original_image
        
        # Initialize predicted image
        predicted = np.zeros_like(s, dtype=np.float64)
        
        # Process each pixel
        for n1 in range(H):
            for n2 in range(W):
                
                # Case 1: First pixel (0,0)
                if n1 == 0 and n2 == 0:
                    predicted[n1, n2] = 0  
                
                # Case 2: First row, interior columns (uses left only)
                elif n1 == 0:
                    predicted[n1, n2] = s[n1, n2 - 1]
                
                # Case 3: First column, interior rows (uses top only)
                elif n2 == 0:
                    predicted[n1, n2] = s[n1 - 1, n2]
                
                # Case 4: Last column (n2 == W-1) - Top-Right neighbor missing
                elif n2 == W - 1:
                    # Redistribute a3 weight among a1, a2, a4
                    total_avail = a1 + a2 + a4
                    if total_avail > 0:
                        w1 = a1 / total_avail
                        w2 = a2 / total_avail
                        w4 = a4 / total_avail
                        predicted[n1, n2] = (
                            w1 * s[n1 - 1, n2 - 1] + 
                            w2 * s[n1 - 1, n2] +
                            w4 * s[n1, n2 - 1]
                        )
                    else:
                        predicted[n1, n2] = s[n1, n2-1] # Fallback
                
                # Case 5: Interior pixels - ALL 4 neighbors available
                else:
                    predicted[n1, n2] = (
                        a1 * s[n1 - 1, n2 - 1] +   # Top-left
                        a2 * s[n1 - 1, n2] +       # Top
                        a3 * s[n1 - 1, n2 + 1] +   # Top-right
                        a4 * s[n1, n2 - 1]         # Left
                    )
        
        self.predicted_image = np.round(predicted).astype(np.int16)
        return self.predicted_image
    
    def compute_error_image(self) -> np.ndarray:
        
        """Compute error (residual) image: e = original - predicted.
        For natural images with strong spatial correlation, most errors cluster around zero,
        resulting in a lower entropy distribution.
        
        Returns:
            np.ndarray: Error image (signed integers, range [-255, +255] for 8-bit)
        """
        
        if self.predicted_image is None:
            self.compute_predicted_image()
        
        self.error_image = self.original_image - self.predicted_image  # Error = Original - Predicted (can be negative)
        
        return self.error_image
    
    
    def compute_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        """Compute intensity histogram for an image.
        Args: image: Input image (can be original or error image)
        Returns: Tuple of (bin_edges, counts):
                - bin_edges: Array of bin boundaries
                - counts: Count of pixels in each bin
        """
        # Flatten image for histogram computation
        flat = image.flatten()
        
        # Determine appropriate bin range
        # For error images, center bins around zero
        # For original images, use full 0-255 range

        min_val, max_val = flat.min(), flat.max()
        
        if min_val < 0:
            # Error image with negative values
            bins = np.arange(min_val, max_val + 2) - 0.5
        else:
            # Original image (0-255 range)
            bins = np.arange(257) - 0.5  # 256 bins for values 0-255
        
        counts, bin_edges = np.histogram(flat, bins=bins)
        
        # Return centers of bins instead of edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers.astype(int), counts
    
    def compute_entropy(self, image: np.ndarray) -> float:
        
        """
        Calculate Shannon entropy: H = -Σ P(i)·log₂(P(i)).
        Args: image: Input image array
        Returns: float: Entropy in bits per pixel
        """
        # Flatten and count occurrences of each intensity value
        flat = image.flatten()
        total_pixels = len(flat)
        
        # Count frequency of each unique value
        value_counts = Counter(flat)
        
        # Compute probabilities
        probabilities = np.array([count / total_pixels for count in value_counts.values()])
        
        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        # Shannon entropy: H = -Σ P(i) * log₂(P(i))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def verify_lossless_reconstruction(self) -> Tuple[bool, float]:
        """
        Verify that original = predicted + error for all pixels.
        Returns: Tuple of (is_lossless, max_error):
                - is_lossless: True if reconstruction is perfect
                - max_error: Maximum absolute reconstruction error
        """
        if self.predicted_image is None or self.error_image is None:
            self.compute_predicted_image()
            self.compute_error_image()
        
        # Reconstruct: predicted + error should equal original
        reconstructed = self.predicted_image + self.error_image
        
        # Calculate reconstruction error
        reconstruction_diff = np.abs(self.original_image - reconstructed)
        max_error = np.max(reconstruction_diff)
        
        # Should be exactly zero for lossless
        is_lossless = max_error < 1e-10
        
        return is_lossless, float(max_error)
    
    def theoretical_compression_ratio(self) -> float:
        
        """Calculate theoretical compression ratio based on Shannon's theorem.
        Returns: float: Theoretical compression ratio (H_original / H_error)
        """
        if self.error_image is None:
            self.compute_error_image()
        
        H_original = self.compute_entropy(self.original_image)
        H_error = self.compute_entropy(self.error_image)
        
        if H_error > 0:
            return H_original / H_error
        else:
            return float('inf')  # Perfect prediction (rare)
    
    def compute_statistics(self) -> Dict:
        
        """Compute comprehensive statistics for the error image.
        Returns: dict: Contains MAE, RMSE, error range, quartiles, etc.
        """
        if self.error_image is None:
            self.compute_error_image()
        
        flat_error = self.error_image.flatten()
        
        stats = {
            'error_min': int(np.min(flat_error)),
            'error_max': int(np.max(flat_error)),
            'error_mean': float(np.mean(flat_error)),
            'error_std': float(np.std(flat_error)),
            'error_mae': float(np.mean(np.abs(flat_error))),  # Mean Absolute Error
            'error_rmse': float(np.sqrt(np.mean(flat_error ** 2))),  # Root Mean Square Error
            'zero_error_count': int(np.sum(flat_error == 0)),
            'zero_error_percentage': float(100.0 * np.sum(flat_error == 0) / len(flat_error)),
            'q1': float(np.percentile(flat_error, 25)),  # First quartile
            'median': float(np.percentile(flat_error, 50)),  # Median
            'q3': float(np.percentile(flat_error, 75)),  # Third quartile
        }
        
        return stats
    
    def visualize_results(self, save_path: Optional[str] = None, dpi: int = 300) -> plt.Figure:
        
        """Create comprehensive 2x3 visualization of results.
        Layout: Row 1: [Original Image] [Predicted Image] [Error Image (offset)]
                Row 2: [Original Histogram] [Error Histogram] [Entropy Comparison]
        Args: save_path: Optional path to save the figure
            dpi: Resolution for saved figure (default 300)
        Returns: matplotlib.figure.Figure: The generated figure
        """
        if self.predicted_image is None or self.error_image is None:
            self.compute_predicted_image()
            self.compute_error_image()
        
        # Compute entropies
        H_orig = self.compute_entropy(self.original_image)
        H_err = self.compute_entropy(self.error_image)
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Lossless Predictive Coding Analysis', fontsize=16, fontweight='bold')
        
        # Row 1, Col 1: Original Image
        ax = axes[0, 0]
        ax.imshow(self.original_image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Original Image\nEntropy: {H_orig:.3f} bits/pixel')
        ax.axis('off')
        
        # Row 1, Col 2: Predicted Image
        ax = axes[0, 1]
        ax.imshow(self.predicted_image, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Predicted Image\n(4-neighbor linear prediction)')
        ax.axis('off')
        
        # Row 1, Col 3: Error Image (with 127 offset for display)
        ax = axes[0, 2]
        error_display = self.error_image + 127  # Offset to center around gray
        error_display = np.clip(error_display, 0, 255)
        im = ax.imshow(error_display, cmap='seismic', vmin=0, vmax=255)
        ax.set_title(f'Error Image (+127 offset)\nEntropy: {H_err:.3f} bits/pixel')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Row 2, Col 1: Original Image Histogram
        ax = axes[1, 0]
        bins_orig, counts_orig = self.compute_histogram(self.original_image)
        ax.bar(bins_orig, counts_orig, width=1, color='steelblue', alpha=0.7)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Original Image Histogram')
        ax.set_xlim(-10, 265)
        ax.grid(True, alpha=0.3)
        
        # Row 2, Col 2: Error Image Histogram
        ax = axes[1, 1]
        bins_err, counts_err = self.compute_histogram(self.error_image)
        ax.bar(bins_err, counts_err, width=1, color='coral', alpha=0.7)
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Image Histogram')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 2, Col 3: Entropy Comparison Bar Chart
        ax = axes[1, 2]
        bars = ax.bar(['Original', 'Error'], [H_orig, H_err], 
                      color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Entropy (bits/pixel)')
        ax.set_title(f'Entropy Comparison\nReduction: {100*(H_orig-H_err)/H_orig:.1f}%')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, [H_orig, H_err]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        
        # Adjustable layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Reserve space for suptitle

        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """
        Returns:
            str: Formatted report with all metrics and analysis
        """
        if self.predicted_image is None or self.error_image is None:
            self.compute_predicted_image()
            self.compute_error_image()
        
        # Compute all metrics
        H_orig = self.compute_entropy(self.original_image)
        H_err = self.compute_entropy(self.error_image)
        is_lossless, max_recon_error = self.verify_lossless_reconstruction()
        theo_cr = self.theoretical_compression_ratio()
        stats = self.compute_statistics()
        
        # Format coefficients
        coeff_str = f"a1={self.coefficients[0]}, a2={self.coefficients[1]}, a3={self.coefficients[2]}, a4={self.coefficients[3]}"
        
        report = f"""
=== LOSSLESS PREDICTIVE CODING ANALYSIS ===
Image: {self.image_path}
Dimensions: {self.height} x {self.width}
Coefficients: {coeff_str}

--- ENTROPY ANALYSIS ---
Original Image (UPLOADED BY USER) Entropy:  {H_orig:.4f} bits/pixel
Error Image Entropy:     {H_err:.4f} bits/pixel
Entropy Reduction:       {100*(H_orig-H_err)/H_orig:.2f}%

--- COMPRESSION POTENTIAL ---
Theoretical CR (Shannon): {theo_cr:.2f}:1
Average bits/pixel (FOR ANY MONOCHROME IMAGE (0-255)): 8.00
Average bits/pixel (error, optimal encoding): {H_err:.2f}

--- ERROR STATISTICS ---
Error Range: [{stats['error_min']}, {stats['error_max']}]
Error Mean: {stats['error_mean']:.4f}
Error Std Dev: {stats['error_std']:.4f}
Mean Absolute Error (MAE): {stats['error_mae']:.4f}
Root Mean Square Error (RMSE): {stats['error_rmse']:.4f}
Error Q1: {stats['q1']:.2f}
Error Median: {stats['median']:.2f}
Error Q3: {stats['q3']:.2f}
Pixels with zero error: {stats['zero_error_count']} ({stats['zero_error_percentage']:.2f}%)

--- VERIFICATION ---
Reconstruction Test: {'PASS' if is_lossless else 'FAIL'}
Max reconstruction error: {max_recon_error}
"""
        return report
    
    def save_outputs(self, output_dir: str = "outputs") -> None:
        """
        Save all output images and reports to specified directory.
        
        Args:
            output_dir: Directory to save outputs (created if doesn't exist)
        
        Saves:
            - predicted_image.png
            - error_image_display.png (with 127 offset)
            - analysis_plots.png
            - statistics_report.txt
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.predicted_image is None or self.error_image is None:
            self.compute_predicted_image()
            self.compute_error_image()
        
        # Save predicted image
        pred_path = os.path.join(output_dir, "predicted_image.png")
        pred_img = Image.fromarray(np.clip(self.predicted_image, 0, 255).astype(np.uint8), mode='L')
        pred_img.save(pred_path)
        print(f"Saved: {pred_path}")
        
        # Save error image with offset for display
        err_display = self.error_image + 127
        err_display = np.clip(err_display, 0, 255).astype(np.uint8)
        err_path = os.path.join(output_dir, "error_image_display.png")
        err_img = Image.fromarray(err_display, mode='L')
        err_img.save(err_path)
        print(f"Saved: {err_path}")
        
        # Save visualization
        plot_path = os.path.join(output_dir, "analysis_plots.png")
        self.visualize_results(save_path=plot_path)
        
        # Save statistics report
        report_path = os.path.join(output_dir, "statistics_report.txt")
        with open(report_path, 'w') as f:
            f.write(self.generate_report())
        print(f"Saved: {report_path}")


def create_test_image() -> np.ndarray:
    
    """Create a simple 4x4 test image with smooth gradient for validation.
    Returns: np.ndarray: 4x4 test image array
    """
    test_image = np.array([
        [100, 105, 110, 108],
        [102, 107, 112, 110],
        [101, 106, 111, 109],
        [103, 108, 113, 111]
    ], dtype=np.uint8)
    return test_image


def create_sample_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    
    """Create a sample grayscale image with gradients and patterns for testing.
    Args: size: Tuple of (height, width)
    Returns: np.ndarray: Synthetic grayscale image
    """
    H, W = size
    image = np.zeros((H, W), dtype=np.uint8)
    
    # Create horizontal gradient
    for x in range(W):
        image[:, x] = int(255 * x / W)
    
    # Add vertical gradient component
    for y in range(H):
        image[y, :] = (image[y, :] * 0.7 + int(255 * y / H) * 0.3).astype(np.uint8)
    
    # Add some structure (diagonal stripes)
    for i in range(min(H, W)):
        if i % 20 < 10:
            image[i, :] = np.clip(image[i, :].astype(np.int16) + 20, 0, 255)
    
    return image


def main():
    """ Main execution function with command-line argument parsing.
        python predictive_coding_q3.py --image path/to/image.png
    """
    parser = argparse.ArgumentParser(
        description='Lossless Predictive Coding for Image Compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python predictive_coding_q3.py --image lena.png
    python predictive_coding_q3.py --output results/
    python predictive_coding_q3.py --test
        '''
    )
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='Path to input grayscale image')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='Output directory for results (default: outputs)')
    parser.add_argument('--test', action='store_true',
                        help='Run with 4x4 test image for validation')
    parser.add_argument('--coefficients', '-c', type=float, nargs=4,
                        default=[0.25, 0.25, 0.25, 0.25],
                        help='Prediction coefficients (a1 a2 a3 a4)')
    
    args = parser.parse_args()
    
    # Determine input source
    if args.test:
        print("Running with 4x4 test image...")
        image_input = create_test_image()
    elif args.image:
        print(f"Loading image: {args.image}")
        image_input = args.image
    else:
        print("No image specified. Using synthetic sample image (256x256)...")
        image_input = create_sample_image()
    
    # Create coder instance
    coefficients = tuple(args.coefficients)
    coder = LosslessPredictiveCoder(image_input, coefficients=coefficients)
    
    # Compute predictions and errors
    print("\nComputing predicted image...")
    coder.compute_predicted_image()
    
    print("Computing error image...")
    coder.compute_error_image()
    
    # Generate and print report
    report = coder.generate_report()
    print(report)
    
    # Save outputs
    print(f"\nSaving outputs to: {args.output}")
    coder.save_outputs(args.output)
    
    print("Done. Check output folder for results.")


if __name__ == "__main__":
    main()
