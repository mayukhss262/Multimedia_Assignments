"""
Huffman Coding for Run-Length Encoded Gray Coded Bit-Plane Images
Each bit plane uses its own Huffman codebook based on individual statistics.
Uses alternating run-length encoding where:
- Rows are assumed to start with 1
- If row starts with 0, first symbol is 0 (zero-length run of 1s)
- Symbols are just run lengths (0, 1, 2, 3, ...) that alternate between 1s and 0s

Examples:
  11000111 -> [2, 3, 3] (2 ones, 3 zeros, 3 ones)
  0001101  -> [0, 3, 2, 1, 1] (0 ones, 3 zeros, 2 ones, 1 zero, 1 one)
"""

import os
import heapq
from collections import Counter
import numpy as np
from PIL import Image


class HuffmanNode:
    """Node for Huffman tree."""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_dict):
    """Build Huffman tree from frequency dictionary."""
    if not freq_dict:
        return None
    
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None


def generate_huffman_codes(root, current_code="", codes=None):
    """Generate Huffman codes from tree."""
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    if root.symbol is not None:
        codes[root.symbol] = current_code if current_code else "0"
        return codes
    
    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)
    
    return codes


def run_length_encode_row(row):
    """
    Performs 1D run-length encoding on a binary row.
    Assumes row starts with 1. If it starts with 0, first symbol is 0.
    Returns list of integer run lengths (alternating between 1s and 0s).
    
    Examples:
      11000111 -> [2, 3, 3] (2 ones, 3 zeros, 3 ones)
      0001101  -> [0, 3, 2, 1, 1] (0 ones, 3 zeros, 2 ones, 1 zero, 1 one)
    """
    binary_row = (row > 0).astype(np.uint8)
    runs = []
    
    if len(binary_row) == 0:
        return runs
    
    # If row starts with 0, add a zero-length run of 1s first
    if binary_row[0] == 0:
        runs.append(0)
    
    current_val = binary_row[0]
    current_run = 1
    
    for pixel in binary_row[1:]:
        if pixel == current_val:
            current_run += 1
        else:
            runs.append(current_run)
            current_val = pixel
            current_run = 1
    
    # Append the last run
    runs.append(current_run)
    
    return runs


def run_length_encode_image(image_array):
    """Performs 1D run-length encoding on entire image."""
    all_runs = []
    for row in image_array:
        row_runs = run_length_encode_row(row)
        all_runs.extend(row_runs)
    return all_runs


def calculate_entropy(freq_dict, total):
    """Calculate entropy of the distribution."""
    entropy = 0
    for count in freq_dict.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def process_individual_codebooks(input_folder, output_file):
    """
    Process gray coded bit plane images with individual Huffman codebooks.
    Each bit plane uses its own statistics and codebook.
    """
    # Find all gray bit plane images
    image_files = sorted([f for f in os.listdir(input_folder) 
                          if f.endswith('.png') and 'gray_bit_plane' in f])
    
    results = []
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("    HUFFMAN CODING FOR RUN-LENGTH ENCODED GRAY CODED BIT-PLANES\n")
        f.write("    (Individual Codebook per Bit Plane)\n")
        f.write("    Alternating Run-Length Encoding (symbols are run lengths: 0,1,2,3,...)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ENCODING SCHEME:\n")
        f.write("  - Rows assumed to start with 1\n")
        f.write("  - If row starts with 0, first symbol is 0 (zero-length run of 1s)\n")
        f.write("  - Symbols alternate: run of 1s, run of 0s, run of 1s, ...\n")
        f.write("  - Example: 11000111 -> [2, 3, 3] (2 ones, 3 zeros, 3 ones)\n")
        f.write("  - Example: 0001101  -> [0, 3, 2, 1, 1]\n\n")
        
        for img_file in image_files:
            img_path = os.path.join(input_folder, img_file)
            
            # Load image
            img = Image.open(img_path).convert('L')
            pixels = np.array(img)
            height, width = pixels.shape
            
            # Uncompressed size (1 bit per pixel)
            uncompressed_bits = height * width
            
            # Get run lengths
            runs = run_length_encode_image(pixels)
            
            # Calculate statistics
            freq_dict = Counter(runs)
            total_symbols = len(runs)
            
            # Build Huffman tree and get codes
            tree = build_huffman_tree(freq_dict)
            huffman_codes = generate_huffman_codes(tree)
            
            # Calculate compressed size using Huffman codes
            huffman_bits = sum(len(huffman_codes[sym]) * freq_dict[sym] for sym in freq_dict)
            
            # Compression ratio
            huffman_ratio = huffman_bits / uncompressed_bits
            
            # Calculate entropy and average code length
            entropy = calculate_entropy(freq_dict, total_symbols)
            avg_code_length = huffman_bits / total_symbols
            
            # Store results
            result = {
                'file': img_file,
                'uncompressed_bits': uncompressed_bits,
                'total_symbols': total_symbols,
                'unique_symbols': len(freq_dict),
                'huffman_bits': huffman_bits,
                'huffman_ratio': huffman_ratio,
                'entropy': entropy,
                'avg_code_length': avg_code_length,
                'huffman_codes': huffman_codes,
                'freq_dict': freq_dict
            }
            results.append(result)
            
            # Write detailed results for this bit plane
            f.write("-" * 80 + "\n")
            f.write(f"Bit Plane: {img_file}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Image size: {width}x{height} = {uncompressed_bits} pixels (bits)\n")
            f.write(f"  Total symbols: {total_symbols}\n")
            f.write(f"  Unique symbols (run lengths): {len(freq_dict)}\n")
            f.write(f"  Entropy: {entropy:.4f} bits/symbol\n")
            f.write(f"  Average Huffman code length: {avg_code_length:.4f} bits/symbol\n")
            f.write(f"\n")
            f.write(f"  Compression Results:\n")
            f.write(f"    Huffman bits:          {huffman_bits:>10}\n")
            f.write(f"    Compression ratio:     {huffman_ratio:.4f}\n")
            f.write(f"\n")
            
            # Complete Huffman codebook (sorted by run length)
            f.write(f"  Huffman Codebook:\n")
            f.write(f"  {'Run Length':<12} {'Count':<10} {'Probability':<12} {'Huffman Code':<25}\n")
            f.write(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*25}\n")
            
            sorted_symbols = sorted(freq_dict.items(), key=lambda x: x[0])
            
            for run_length, count in sorted_symbols:
                prob = count / total_symbols
                code = huffman_codes[run_length]
                f.write(f"  {run_length:<12} {count:<10} {prob:<12.6f} {code:<25}\n")
            
            f.write("\n")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Bit Plane':<25} {'Huffman Ratio':<15} {'Entropy':<12} {'Avg Code Len':<12}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            bit_num = r['file'].split('_')[-1].replace('.png', '')
            f.write(f"gray_bit_plane_{bit_num:<10} {r['huffman_ratio']:<15.4f} "
                    f"{r['entropy']:<12.4f} {r['avg_code_length']:<12.4f}\n")
        
        f.write("-" * 80 + "\n")
        
        # Averages
        avg_huffman = sum(r['huffman_ratio'] for r in results) / len(results)
        avg_entropy = sum(r['entropy'] for r in results) / len(results)
        avg_code = sum(r['avg_code_length'] for r in results) / len(results)
        
        f.write(f"{'Average':<25} {avg_huffman:<15.4f} "
                f"{avg_entropy:<12.4f} {avg_code:<12.4f}\n")
        f.write("\n")
        f.write("Note: Compression ratio = compressed bits / original bits\n")
        f.write("      Ratio < 1 means compression, Ratio > 1 means expansion\n")
    
    print(f"Results saved to: {output_file}")
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'gray_coded_bit_plane_images')
    output_file = os.path.join(script_dir, '2b_huffman_individual_results.txt')
    
    if not os.path.exists(input_folder):
        print(f"Error: {input_folder} not found!")
        print("Please run 1b_generate_graycode_images.py first.")
        return
    
    print("=" * 60)
    print("Huffman Coding with Individual Codebooks per Bit Plane")
    print("(Alternating Run-Length Encoding)")
    print("=" * 60)
    
    results = process_individual_codebooks(input_folder, output_file)
    
    # Print summary
    print("\nSummary:")
    print(f"{'Bit Plane':<20} {'Huffman Ratio':<15}")
    print("-" * 40)
    for r in results:
        bit_num = r['file'].split('_')[-1].replace('.png', '')
        print(f"Bit {bit_num:<15} {r['huffman_ratio']:<15.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
