"""
Huffman Coding for Run-Length Encoded Gray Coded Bit-Plane Images
Each bit plane uses its own Huffman codebook based on individual statistics.
Distinguishes between runs of 0s and runs of 1s (separate symbols).
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
    
    # Create leaf nodes
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    
    # Build tree
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
    
    # Leaf node
    if root.symbol is not None:
        codes[root.symbol] = current_code if current_code else "0"
        return codes
    
    # Traverse left and right
    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)
    
    return codes


def run_length_encode_row_with_values(row):
    """
    Performs 1D run-length encoding on a binary row.
    Returns list of tuples: (value, run_length) where value is 0 or 1.
    Rows are assumed to start with 1. If row starts with 0, first run is (1, 0).
    No cap on run length - runs can be any length.
    """
    binary_row = (row > 0).astype(np.uint8)
    runs = []
    
    if len(binary_row) == 0:
        return runs
    
    if binary_row[0] == 0:
        # Row starts with 0, so there's a zero-length run of 1s first
        runs.append((1, 0))
    
    current_val = binary_row[0]
    current_run = 0
    
    for pixel in binary_row:
        if pixel == current_val:
            current_run += 1
        else:
            runs.append((current_val, current_run))
            current_val = pixel
            current_run = 1
    
    if current_run > 0:
        runs.append((current_val, current_run))
    
    return runs


def run_length_encode_image_with_values(image_array):
    """Performs 1D run-length encoding on entire image, tracking value type."""
    all_runs = []
    for row in image_array:
        row_runs = run_length_encode_row_with_values(row)
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
    Runs of 0s and 1s are treated as separate symbols.
    """
    # Find all gray bit plane images
    image_files = sorted([f for f in os.listdir(input_folder) 
                          if f.endswith('.png') and 'gray_bit_plane' in f])
    
    results = []
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("    HUFFMAN CODING FOR RUN-LENGTH ENCODED GRAY CODED BIT-PLANES\n")
        f.write("    (Individual Codebook per Bit Plane)\n")
        f.write("    Separate symbols for runs of 0s and runs of 1s\n")
        f.write("=" * 80 + "\n\n")
        
        for img_file in image_files:
            img_path = os.path.join(input_folder, img_file)
            
            # Load image
            img = Image.open(img_path).convert('L')
            pixels = np.array(img)
            height, width = pixels.shape
            
            # Uncompressed size (1 bit per pixel)
            uncompressed_bits = height * width
            
            # Get run lengths with value information
            runs = run_length_encode_image_with_values(pixels)
            
            # Create symbols: "0_<length>" for runs of 0s, "1_<length>" for runs of 1s
            symbols = [f"{val}_{length}" for val, length in runs]
            
            # Calculate statistics
            freq_dict = Counter(symbols)
            total_runs = len(symbols)
            
            # Calculate probabilities
            prob_dict = {k: v / total_runs for k, v in freq_dict.items()}
            
            # Build Huffman tree and get codes
            tree = build_huffman_tree(freq_dict)
            huffman_codes = generate_huffman_codes(tree)
            
            # Calculate compressed size using Huffman codes
            huffman_bits = sum(len(huffman_codes[sym]) * freq_dict[sym] for sym in freq_dict)
            
            # Fixed 6-bit encoding for comparison (value is implicit from alternation)
            fixed_bits = total_runs * 6  # 6 bits for length, value alternates
            
            # Compression ratios
            fixed_ratio = fixed_bits / uncompressed_bits
            huffman_ratio = huffman_bits / uncompressed_bits
            
            # Calculate entropy and average code length
            entropy = calculate_entropy(freq_dict, total_runs)
            avg_code_length = huffman_bits / total_runs
            
            # Store results
            result = {
                'file': img_file,
                'uncompressed_bits': uncompressed_bits,
                'total_runs': total_runs,
                'unique_symbols': len(freq_dict),
                'fixed_bits': fixed_bits,
                'huffman_bits': huffman_bits,
                'fixed_ratio': fixed_ratio,
                'huffman_ratio': huffman_ratio,
                'entropy': entropy,
                'avg_code_length': avg_code_length,
                'huffman_codes': huffman_codes,
                'prob_dict': prob_dict
            }
            results.append(result)
            
            # Write detailed results for this bit plane
            f.write("-" * 80 + "\n")
            f.write(f"Bit Plane: {img_file}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Image size: {width}x{height} = {uncompressed_bits} pixels (bits)\n")
            f.write(f"  Total runs: {total_runs}\n")
            f.write(f"  Unique symbols (value, length pairs): {len(freq_dict)}\n")
            f.write(f"  Entropy: {entropy:.4f} bits/symbol\n")
            f.write(f"  Average Huffman code length: {avg_code_length:.4f} bits/symbol\n")
            f.write(f"\n")
            f.write(f"  Compression Results:\n")
            f.write(f"    Fixed 6-bit RLE:       {fixed_bits:>10} bits, ratio = {fixed_ratio:.4f}\n")
            f.write(f"    Huffman + RLE:         {huffman_bits:>10} bits, ratio = {huffman_ratio:.4f}\n")
            f.write(f"    Improvement:           {(fixed_ratio - huffman_ratio) / fixed_ratio * 100:.2f}%\n")
            f.write(f"\n")
            
            # Complete Huffman codebook (sorted by symbol)
            f.write(f"  Complete Huffman Codebook:\n")
            f.write(f"  {'Symbol':<12} {'Value':<8} {'Length':<8} {'Count':<10} {'Prob':<10} {'Huffman Code':<25}\n")
            f.write(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*25}\n")
            
            # Sort by value first, then by run length
            def sort_key(item):
                sym = item[0]
                parts = sym.split('_')
                return (int(parts[0]), int(parts[1]))
            
            sorted_symbols = sorted(freq_dict.items(), key=sort_key)
            
            # First show runs of 0s
            f.write(f"\n  --- Runs of 0s (black) ---\n")
            for sym, count in sorted_symbols:
                val, length = sym.split('_')
                if val == '0':
                    prob = count / total_runs
                    code = huffman_codes[sym]
                    f.write(f"  {sym:<12} {val:<8} {length:<8} {count:<10} {prob:<10.6f} {code:<25}\n")
            
            # Then show runs of 1s
            f.write(f"\n  --- Runs of 1s (white) ---\n")
            for sym, count in sorted_symbols:
                val, length = sym.split('_')
                if val == '1':
                    prob = count / total_runs
                    code = huffman_codes[sym]
                    f.write(f"  {sym:<12} {val:<8} {length:<8} {count:<10} {prob:<10.6f} {code:<25}\n")
            
            f.write("\n")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Bit Plane':<25} {'Fixed 6-bit':<12} {'Huffman+RLE':<12} {'Entropy':<10} {'Avg Code':<10}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            bit_num = r['file'].split('_')[-1].replace('.png', '')
            f.write(f"gray_bit_plane_{bit_num:<10} {r['fixed_ratio']:<12.4f} {r['huffman_ratio']:<12.4f} "
                    f"{r['entropy']:<10.4f} {r['avg_code_length']:<10.4f}\n")
        
        f.write("-" * 80 + "\n")
        
        # Averages
        avg_fixed = sum(r['fixed_ratio'] for r in results) / len(results)
        avg_huffman = sum(r['huffman_ratio'] for r in results) / len(results)
        avg_entropy = sum(r['entropy'] for r in results) / len(results)
        avg_code = sum(r['avg_code_length'] for r in results) / len(results)
        
        f.write(f"{'Average':<25} {avg_fixed:<12.4f} {avg_huffman:<12.4f} "
                f"{avg_entropy:<10.4f} {avg_code:<10.4f}\n")
        f.write("\n")
        f.write(f"Overall improvement with Huffman: {(avg_fixed - avg_huffman) / avg_fixed * 100:.2f}%\n")
        f.write("\nNote: Ratio < 1 means compression, Ratio > 1 means expansion.\n")
        f.write("\nSymbol format: <value>_<run_length>\n")
        f.write("  - 0_5 means a run of 5 consecutive 0s (black pixels)\n")
        f.write("  - 1_3 means a run of 3 consecutive 1s (white pixels)\n")
    
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
    print("(Separate symbols for runs of 0s and runs of 1s)")
    print("=" * 60)
    
    results = process_individual_codebooks(input_folder, output_file)
    
    # Print summary
    print("\nSummary:")
    print(f"{'Bit Plane':<20} {'Fixed 6-bit':<12} {'Huffman+RLE':<12}")
    print("-" * 50)
    for r in results:
        bit_num = r['file'].split('_')[-1].replace('.png', '')
        print(f"Bit {bit_num:<15} {r['fixed_ratio']:<12.4f} {r['huffman_ratio']:<12.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
