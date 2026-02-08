import os
import subprocess

def load_pgm(filepath):
    """
    Reads a PGM P2 file (ASCII) and returns (width, height, maxval, pixels).
    pixels is a list of integers.
    """
    with open(filepath, 'r') as f:
        # Read lines and remove comments
        lines = [line.strip() for line in f if not line.strip().startswith('#')]
        
    # Flatten the list of strings into a single list of tokens
    tokens = []
    for line in lines:
        tokens.extend(line.split())
        
    if tokens[0] != 'P2':
        raise ValueError("Not a P2 PGM file")
        
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])
    pixels = [int(t) for t in tokens[4:]]
    
    return width, height, maxval, pixels

def save_pgm(filepath, width, height, maxval, pixels):
    """
    Writes a PGM P2 file.
    """
    with open(filepath, 'w') as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write(f"{maxval}\n")
        
        # Write pixels, typically roughly 70 chars per line max, but we can just write one per line or a few
        # To affect speed less, let's write them cleanly.
        line_buffer = []
        for i, p in enumerate(pixels):
            line_buffer.append(str(p))
            if len(line_buffer) >= 15: # modest batch
                f.write(" ".join(line_buffer) + "\n")
                line_buffer = []
        if line_buffer:
            f.write(" ".join(line_buffer) + "\n")

def get_bit_plane(pixels, bit_position):
    """
    Extracts the bit plane.
    Returns: list of pixels (0 or 255)
    """
    mask = 1 << bit_position
    return [255 if (p & mask) else 0 for p in pixels]

def binary_to_gray(pixels):
    """
    Converts list of pixels to Gray code.
    g = b XOR (b >> 1)
    """
    return [(p ^ (p >> 1)) for p in pixels]

def main():
    input_image = 'image_grayscale.jpeg'
    temp_pgm = 'temp_input.pgm'
    
    if not os.path.exists(input_image):
        print(f"Error: {input_image} not found.")
        return

    print("Converting image to PGM...")
    # Convert JPEG to P2 PGM (ASCII)
    subprocess.run(['convert', input_image, '-compress', 'none', temp_pgm], check=True)
    
    print("Loading PGM...")
    width, height, maxval, pixels = load_pgm(temp_pgm)
    print(f"Loaded {width}x{height} image.")
    
    output_dir = "output_bit_planes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Part (a): Bit-plane slicing
    print("\n--- Part (a): Bit-plane slicing of Binary code ---")
    for i in range(8):
        plane_pixels = get_bit_plane(pixels, i)
        pgm_filename = os.path.join(output_dir, f"binary_bit_plane_{i}.pgm")
        png_filename = os.path.join(output_dir, f"binary_bit_plane_{i}.png")
        save_pgm(pgm_filename, width, height, 255, plane_pixels)
        subprocess.run(['convert', pgm_filename, png_filename], check=True)
        os.remove(pgm_filename)
        print(f"Saved {png_filename}")

    # Part (b): Gray code conversion and slicing
    print("\n--- Part (b): Bit-plane slicing of Gray code ---")
    gray_pixels = binary_to_gray(pixels)
    
    # Save full gray code image
    gray_pgm = os.path.join(output_dir, "gray_code_image.pgm")
    gray_png = os.path.join(output_dir, "gray_code_image.png")
    save_pgm(gray_pgm, width, height, maxval, gray_pixels)
    subprocess.run(['convert', gray_pgm, gray_png], check=True)
    os.remove(gray_pgm)
    print(f"Saved {gray_png}")
    
    for i in range(8):
        plane_pixels = get_bit_plane(gray_pixels, i)
        pgm_filename = os.path.join(output_dir, f"gray_bit_plane_{i}.pgm")
        png_filename = os.path.join(output_dir, f"gray_bit_plane_{i}.png")
        save_pgm(pgm_filename, width, height, 255, plane_pixels)
        subprocess.run(['convert', pgm_filename, png_filename], check=True)
        os.remove(pgm_filename)
        print(f"Saved {png_filename}")

    # Cleanup input temp file
    if os.path.exists(temp_pgm):
        os.remove(temp_pgm)

    # Part (c): Comparison
    print("\n--- Part (c): Comparison and Justification ---")
    observation = """
    Observation:
    When comparing the bit-planes of the standard binary representation (Binary Code) 
    versus the Gray code representation:
    
    1. Higher bit planes (MSB, e.g., bit 7, 6) typically contain the most significant visual 
       structure in both representations.
    2. Lower bit planes (LSB, e.g., bit 0, 1) in pure Binary Code often appear as random noise. 
       This is because adjacent pixel values often change in the LSB even for small intensity variations.
    3. Gray Complexity: In standard Binary Code, a change from 127 (01111111) to 128 (10000000) 
       flips ALL 8 bits. In Gray Code, adjacent values only differ by exactly one bit.
    
    Why Gray-coded bit-planes are better:
    - Correlation: Gray code preserves more inter-pixel correlation in lower bit planes compared 
      to pure binary. In binary, a smooth gradient can cause rapid oscillation in lower bit planes 
      (e.g., 7->8 is 0111->1000, flipping 4 bits). In Gray code, it flips only 1 bit.
    - Compression: Because Gray code bit-planes (especially higher ones, but also lower ones to 
      an extent) tend to have larger uniform areas (fewer 0->1 transitions for gradual intensity 
      changes), they are more compressible using algorithms like Run-Length Encoding bit-plane by bit-plane.
    """
    print(observation)
    with open(os.path.join(output_dir, "observation.txt"), "w") as f:
        f.write(observation)

if __name__ == "__main__":
    main()
