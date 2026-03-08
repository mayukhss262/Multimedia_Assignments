import sys
import math

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

def delta_modulation(img_array, zeta, alpha=1.0):
    rows, cols = img_array.shape
    reconstructed = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        prev_reconstructed = 128.0 # Starting prediction at mid-gray
        for c in range(cols):
            # prediction
            prediction = alpha * prev_reconstructed
            
            # error
            error = img_array[r, c] - prediction
            
            # quantize
            if error >= 0:
                quantized_error = zeta
            else:
                quantized_error = -zeta
                
            # reconstruct
            recon_pixel = prediction + quantized_error
            
            # bound
            if recon_pixel > 255:
                recon_pixel = 255
            elif recon_pixel < 0:
                recon_pixel = 0
                
            reconstructed[r, c] = recon_pixel
            prev_reconstructed = recon_pixel
            
    return reconstructed

def main():
    img_path = 'lena.png'
    try:
        img = Image.open(img_path).convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        return
        
    img_array = np.array(img, dtype=np.float32)
    zetas = [5, 10, 20]
    
    # Define a variable alpha which can be changed as wanted
    alpha = 1.0 
    
    psnr_results = {}
    
    for zeta in zetas:
        recon_array = delta_modulation(img_array, zeta, alpha)
        
        # Calculate PSNR
        mse = np.mean((img_array - recon_array) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * math.log10((255.0 ** 2) / mse)
            
        psnr_results[zeta] = psnr
        
        out_img = Image.fromarray(recon_array.astype(np.uint8))
        out_img.save(f'reconstructed_zeta_{zeta}.png')
        print(f"Saved reconstructed_zeta_{zeta}.png with PSNR: {psnr:.2f} dB")
        
    with open('part_1_dm_results.txt', 'w') as f:
        f.write("Delta Modulation Assignment Solution\n")
        f.write("--------------------------------------\n")
        f.write(f"Parameters used: alpha = {alpha}\n\n")
        f.write("1. (b) Reconstructed Image Observations:\n")
        f.write("   - For zeta = 5 (Small step size): Slope overload is prominently observed. The reconstruction cannot track rapid changes (such as edges) quickly enough, resulting in blurred edges.\n")
        f.write("   - For zeta = 10 (Medium step size): An intermediate quality that balances granular noise and slope overload.\n")
        f.write("   - For zeta = 20 (Large step size): Granular noise is prominently observed. In smooth regions of the image, the reconstructed signal oscillates around the true value with a large amplitude, giving a grainy appearance.\n\n")
        
        f.write("1. (c) PSNR of reconstructed images:\n")
        for zeta in zetas:
            f.write(f"   - PSNR for zeta = {zeta}: {psnr_results[zeta]:.2f} dB\n")

if __name__ == "__main__":
    main()
