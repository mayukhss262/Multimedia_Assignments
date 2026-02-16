"""
Huffman Coding with Common Codebook for All Gray Coded Bit-Planes
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


def process_common_codebook(input_folder, output_file):
    """
    Process all gray coded bit plane images with a common Huffman codebook.
    Statistics are gathered from all 8 bit planes combined.
    """
    # Find all gray bit plane images
    image_files = sorted([f for f in os.listdir(input_folder) 
                          if f.endswith('.png') and 'gray_bit_plane' in f])
    
    # First pass: gather statistics from all images
    print("Phase 1: Gathering statistics from all bit planes...")
    all_symbols = []
    image_data = {}  # Store runs per image for later
    
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        
        img = Image.open(img_path).convert('L')
        pixels = np.array(img)
        
        runs = run_length_encode_image(pixels)
        
        image_data[img_file] = {
            'pixels': pixels,
            'runs': runs
        }
        all_symbols.extend(runs)
        print(f"  {img_file}: {len(runs)} symbols")
    
    # Calculate combined statistics
    print("\nPhase 2: Building common Huffman codebook...")
    combined_freq = Counter(all_symbols)
    total_symbols = len(all_symbols)
    
    # Build common Huffman tree
    tree = build_huffman_tree(combined_freq)
    common_codes = generate_huffman_codes(tree)
    
    # Calculate entropy
    entropy = calculate_entropy(combined_freq, total_symbols)
    
    print(f"  Total symbols across all planes: {total_symbols}")
    print(f"  Unique symbols (run lengths): {len(combined_freq)}")
    print(f"  Combined entropy: {entropy:.4f} bits/symbol")
    
    # Second pass: apply common codebook to each image
    print("\nPhase 3: Encoding each bit plane with common codebook...")
    results = []
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("    HUFFMAN CODING WITH COMMON CODEBOOK FOR ALL GRAY CODED BIT-PLANES\n")
        f.write("    Alternating Run-Length Encoding (symbols are run lengths: 0,1,2,3,...)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ENCODING SCHEME:\n")
        f.write("  - Rows assumed to start with 1\n")
        f.write("  - If row starts with 0, first symbol is 0 (zero-length run of 1s)\n")
        f.write("  - Symbols alternate: run of 1s, run of 0s, run of 1s, ...\n")
        f.write("  - Example: 11000111 -> [2, 3, 3] (2 ones, 3 zeros, 3 ones)\n")
        f.write("  - Example: 0001101  -> [0, 3, 2, 1, 1]\n\n")
        
        # Write common codebook first
        f.write("-" * 80 + "\n")
        f.write("COMMON HUFFMAN CODEBOOK (shared across all bit planes)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total symbols across all planes: {total_symbols}\n")
        f.write(f"Unique symbols (run lengths): {len(combined_freq)}\n")
        f.write(f"Combined entropy: {entropy:.4f} bits/symbol\n\n")
        
        f.write(f"  {'Run Length':<12} {'Count':<12} {'Probability':<12} {'Huffman Code':<30}\n")
        f.write(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*30}\n")
        
        # Sort by run length
        sorted_symbols = sorted(combined_freq.items(), key=lambda x: x[0])
        
        for run_length, count in sorted_symbols:
            prob = count / total_symbols
            code = common_codes[run_length]
            f.write(f"  {run_length:<12} {count:<12} {prob:<12.6f} {code:<30}\n")
        
        f.write("\n")
        
        # Now show results for each bit plane
        f.write("=" * 80 + "\n")
        f.write("COMPRESSION RESULTS PER BIT PLANE (using common codebook)\n")
        f.write("=" * 80 + "\n\n")
        
        for img_file in image_files:
            data = image_data[img_file]
            runs = data['runs']
            pixels = data['pixels']
            height, width = pixels.shape
            
            uncompressed_bits = height * width
            num_symbols = len(runs)
            
            # Huffman with common codebook
            huffman_bits = sum(len(common_codes[sym]) for sym in runs)
            
            huffman_ratio = huffman_bits / uncompressed_bits
            avg_code_length = huffman_bits / num_symbols
            
            result = {
                'file': img_file,
                'uncompressed_bits': uncompressed_bits,
                'num_symbols': num_symbols,
                'huffman_bits': huffman_bits,
                'huffman_ratio': huffman_ratio,
                'avg_code_length': avg_code_length
            }
            results.append(result)
            
            f.write(f"{img_file}:\n")
            f.write(f"  Symbols: {num_symbols}, Huffman bits: {huffman_bits}\n")
            f.write(f"  Compression ratio: {huffman_ratio:.4f}\n")
            f.write(f"  Average code length: {avg_code_length:.4f} bits/symbol\n\n")
            
            print(f"  {img_file}: Ratio={huffman_ratio:.4f}")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Bit Plane':<25} {'Symbols':<12} {'Huffman Ratio':<15} {'Avg Code Len':<12}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            bit_num = r['file'].split('_')[-1].replace('.png', '')
            f.write(f"gray_bit_plane_{bit_num:<10} {r['num_symbols']:<12} {r['huffman_ratio']:<15.4f} "
                    f"{r['avg_code_length']:<12.4f}\n")
        
        f.write("-" * 80 + "\n")
        
        # Averages
        avg_huffman = sum(r['huffman_ratio'] for r in results) / len(results)
        avg_code = sum(r['avg_code_length'] for r in results) / len(results)
        
        f.write(f"{'Average':<25} {'':<12} {avg_huffman:<15.4f} {avg_code:<12.4f}\n")
        f.write("\n")
        f.write("Note: Compression ratio = compressed bits / original bits\n")
        f.write("      Ratio < 1 means compression, Ratio > 1 means expansion\n")
    
    print(f"\nResults saved to: {output_file}")
    return results, common_codes


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'gray_coded_bit_plane_images')
    output_file = os.path.join(script_dir, '2b_huffman_common_results.txt')
    
    if not os.path.exists(input_folder):
        print(f"Error: {input_folder} not found!")
        print("Please run 1b_generate_graycode_images.py first.")
        return
    
    print("=" * 60)
    print("Huffman Coding with Common Codebook Across All Bit Planes")
    print("(Alternating Run-Length Encoding)")
    print("=" * 60)
    
    results, common_codes = process_common_codebook(input_folder, output_file)
    
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
