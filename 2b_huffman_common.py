"""
Huffman Coding with Common Codebook for All Gray Coded Bit-Planes
Generates statistics across all 8 bit planes combined,
then creates a single shared Huffman codebook for encoding all images.
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


def run_length_encode_row_with_values(row):
    """
    Performs 1D run-length encoding on a binary row.
    Returns list of tuples: (value, run_length).
    """
    binary_row = (row > 0).astype(np.uint8)
    runs = []
    
    if len(binary_row) == 0:
        return runs
    
    if binary_row[0] == 0:
        runs.append((1, 0))
    
    current_val = binary_row[0]
    current_run = 0
    
    for pixel in binary_row:
        if pixel == current_val:
            current_run += 1
            if current_run > 63:
                runs.append((current_val, 63))
                runs.append((1 - current_val, 0))
                current_run = 1
        else:
            runs.append((current_val, min(current_run, 63)))
            current_val = pixel
            current_run = 1
    
    if current_run > 0:
        runs.append((current_val, min(current_run, 63)))
    
    return runs


def run_length_encode_image_with_values(image_array):
    """Performs 1D run-length encoding on entire image."""
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
        
        runs = run_length_encode_image_with_values(pixels)
        symbols = [f"{val}_{length}" for val, length in runs]
        
        image_data[img_file] = {
            'pixels': pixels,
            'runs': runs,
            'symbols': symbols
        }
        all_symbols.extend(symbols)
        print(f"  {img_file}: {len(symbols)} runs")
    
    # Calculate combined statistics
    print("\nPhase 2: Building common Huffman codebook...")
    combined_freq = Counter(all_symbols)
    total_runs = len(all_symbols)
    
    # Build common Huffman tree
    tree = build_huffman_tree(combined_freq)
    common_codes = generate_huffman_codes(tree)
    
    # Calculate entropy
    entropy = calculate_entropy(combined_freq, total_runs)
    
    print(f"  Total runs across all planes: {total_runs}")
    print(f"  Unique symbols: {len(combined_freq)}")
    print(f"  Combined entropy: {entropy:.4f} bits/symbol")
    
    # Second pass: apply common codebook to each image
    print("\nPhase 3: Encoding each bit plane with common codebook...")
    results = []
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("    HUFFMAN CODING WITH COMMON CODEBOOK FOR ALL GRAY CODED BIT-PLANES\n")
        f.write("    Statistics gathered from all 8 bit planes combined\n")
        f.write("=" * 80 + "\n\n")
        
        # Write common codebook first
        f.write("-" * 80 + "\n")
        f.write("COMMON HUFFMAN CODEBOOK (shared across all bit planes)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total runs across all planes: {total_runs}\n")
        f.write(f"Unique symbols: {len(combined_freq)}\n")
        f.write(f"Combined entropy: {entropy:.4f} bits/symbol\n\n")
        
        f.write(f"  {'Symbol':<12} {'Value':<8} {'Length':<8} {'Count':<10} {'Prob':<10} {'Huffman Code':<25}\n")
        f.write(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*25}\n")
        
        # Sort by value first, then by run length
        def sort_key(item):
            sym = item[0]
            parts = sym.split('_')
            return (int(parts[0]), int(parts[1]))
        
        sorted_symbols = sorted(combined_freq.items(), key=sort_key)
        
        # Runs of 0s
        f.write(f"\n  --- Runs of 0s (black) ---\n")
        for sym, count in sorted_symbols:
            val, length = sym.split('_')
            if val == '0':
                prob = count / total_runs
                code = common_codes[sym]
                f.write(f"  {sym:<12} {val:<8} {length:<8} {count:<10} {prob:<10.6f} {code:<25}\n")
        
        # Runs of 1s
        f.write(f"\n  --- Runs of 1s (white) ---\n")
        for sym, count in sorted_symbols:
            val, length = sym.split('_')
            if val == '1':
                prob = count / total_runs
                code = common_codes[sym]
                f.write(f"  {sym:<12} {val:<8} {length:<8} {count:<10} {prob:<10.6f} {code:<25}\n")
        
        f.write("\n")
        
        # Now show results for each bit plane
        f.write("=" * 80 + "\n")
        f.write("COMPRESSION RESULTS PER BIT PLANE (using common codebook)\n")
        f.write("=" * 80 + "\n\n")
        
        for img_file in image_files:
            data = image_data[img_file]
            symbols = data['symbols']
            pixels = data['pixels']
            height, width = pixels.shape
            
            uncompressed_bits = height * width
            num_runs = len(symbols)
            
            # Fixed 6-bit RLE
            fixed_bits = num_runs * 6
            
            # Huffman with common codebook
            huffman_bits = sum(len(common_codes[sym]) for sym in symbols)
            
            fixed_ratio = fixed_bits / uncompressed_bits
            huffman_ratio = huffman_bits / uncompressed_bits
            avg_code_length = huffman_bits / num_runs
            
            result = {
                'file': img_file,
                'uncompressed_bits': uncompressed_bits,
                'num_runs': num_runs,
                'fixed_bits': fixed_bits,
                'huffman_bits': huffman_bits,
                'fixed_ratio': fixed_ratio,
                'huffman_ratio': huffman_ratio,
                'avg_code_length': avg_code_length
            }
            results.append(result)
            
            f.write(f"{img_file}:\n")
            f.write(f"  Runs: {num_runs}, Fixed 6-bit: {fixed_bits} bits, Huffman: {huffman_bits} bits\n")
            f.write(f"  Fixed ratio: {fixed_ratio:.4f}, Huffman ratio: {huffman_ratio:.4f}\n")
            f.write(f"  Improvement: {(fixed_ratio - huffman_ratio) / fixed_ratio * 100:.2f}%\n\n")
            
            print(f"  {img_file}: Fixed={fixed_ratio:.4f}, Huffman={huffman_ratio:.4f}")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Bit Plane':<25} {'Fixed 6-bit':<12} {'Huffman+RLE':<12} {'Avg Code Len':<12}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            bit_num = r['file'].split('_')[-1].replace('.png', '')
            f.write(f"gray_bit_plane_{bit_num:<10} {r['fixed_ratio']:<12.4f} {r['huffman_ratio']:<12.4f} "
                    f"{r['avg_code_length']:<12.4f}\n")
        
        f.write("-" * 80 + "\n")
        
        # Averages
        avg_fixed = sum(r['fixed_ratio'] for r in results) / len(results)
        avg_huffman = sum(r['huffman_ratio'] for r in results) / len(results)
        avg_code = sum(r['avg_code_length'] for r in results) / len(results)
        
        f.write(f"{'Average':<25} {avg_fixed:<12.4f} {avg_huffman:<12.4f} {avg_code:<12.4f}\n")
        f.write("\n")
        f.write(f"Overall improvement with common Huffman codebook: {(avg_fixed - avg_huffman) / avg_fixed * 100:.2f}%\n")
        f.write("\nNote: Using a common codebook is less optimal per-image, but requires\n")
        f.write("storing only ONE codebook instead of 8 separate ones.\n")
    
    print(f"\nResults saved to: {output_file}")
    return results, common_codes


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'gray_coded_bit_plane_images')
    output_file = os.path.join(script_dir, '2c_huffman_common_results.txt')
    
    if not os.path.exists(input_folder):
        print(f"Error: {input_folder} not found!")
        print("Please run 1b_generate_graycode_images.py first.")
        return
    
    print("=" * 60)
    print("Huffman Coding with Common Codebook Across All Bit Planes")
    print("=" * 60)
    
    results, common_codes = process_common_codebook(input_folder, output_file)
    
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
