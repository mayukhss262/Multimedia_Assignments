import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def get_pixel(img_array, r, c):
    rows, cols = img_array.shape
    r = max(0, min(r, rows - 1))
    c = max(0, min(c, cols - 1))
    return img_array[r, c]

def predict_open_loop(img_array):
    rows, cols = img_array.shape
    predicted = np.zeros_like(img_array, dtype=np.float32)
    error = np.zeros_like(img_array, dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            # Causal neighbors
            left = get_pixel(img_array, r, c - 1)
            top = get_pixel(img_array, r - 1, c)
            top_left = get_pixel(img_array, r - 1, c - 1)
            top_right = get_pixel(img_array, r - 1, c + 1)
            
            # Predict
            pred_val = 0.4 * left + 0.4 * top + 0.1 * top_left + 0.1 * top_right
            predicted[r, c] = pred_val
            error[r, c] = img_array[r, c] - pred_val
            
    return predicted, error

def get_lloyd_max_table_unit_variance_laplacian(levels):
    if levels == 2:
        thresholds = [0.0]
        reconstruction = [-0.7071, 0.7071]
    elif levels == 4:
        thresholds = [-1.102, 0.0, 1.102]
        reconstruction = [-1.834, -0.395, 0.395, 1.834]
        # Or using the search found ones: [-1.13, 0, 1.13] and [-1.83, -0.42, 0.42, 1.83]
        # We will use the standard from literature. 
        # For M=4 laplacian, t=[-1.102, 0, 1.102] and y=[-1.834, -0.395, 0.395, 1.834]
    elif levels == 8:
        thresholds = [-2.3796, -1.2527, -0.5332, 0.0, 0.5332, 1.2527, 2.3796]
        reconstruction = [-3.0867, -1.6725, -0.8330, -0.2334, 0.2334, 0.8330, 1.6725, 3.0867]
    else:
        raise ValueError("Unsupported number of levels")
        
    return np.array(thresholds), np.array(reconstruction)

def quantize(value, thresholds, reconstruction):
    for i, t in enumerate(thresholds):
        if value < t:
            return reconstruction[i]
    return reconstruction[-1]

def dpcm(img_array, thresholds, reconstruction):
    rows, cols = img_array.shape
    reconstructed_img = np.zeros_like(img_array, dtype=np.float32)
    error_q_img = np.zeros_like(img_array, dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            left = get_pixel(reconstructed_img, r, c - 1) if c > 0 else 128.0
            top = get_pixel(reconstructed_img, r - 1, c) if r > 0 else 128.0
            top_left = get_pixel(reconstructed_img, r - 1, c - 1) if (r > 0 and c > 0) else 128.0
            top_right = get_pixel(reconstructed_img, r - 1, c + 1) if (r > 0 and c < cols - 1) else 128.0
            
            # Predict
            pred_val = 0.4 * left + 0.4 * top + 0.1 * top_left + 0.1 * top_right
            
            # Error
            err = img_array[r, c] - pred_val
            
            # Quantize error
            err_q = quantize(err, thresholds, reconstruction)
            error_q_img[r, c] = err_q
            
            # Reconstruct pixel
            recon_pixel = pred_val + err_q
            
            # Bound
            if recon_pixel > 255: recon_pixel = 255
            elif recon_pixel < 0: recon_pixel = 0
                
            reconstructed_img[r, c] = recon_pixel
            
    return reconstructed_img

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255.0 ** 2 / mse)

def main():
    img_path = 'lena.png'
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # 2(a) and 2(b): Open-loop prediction and error image
    predicted, error = predict_open_loop(img_array)
    
    # Save error image (scaled for visibility)
    # Scale error from [-255, 255] to [0, 255]
    error_display = np.clip((error + 255) / 2, 0, 255).astype(np.uint8)
    Image.fromarray(error_display).save('dpcm_error_image.png')
    
    # Plot histogram
    plt.figure()
    plt.hist(error.flatten(), bins=100, range=(-100, 100), color='gray', alpha=0.7)
    plt.title('Error Image Histogram')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig('dpcm_error_histogram.png')
    plt.close()
    
    # 2(c): Error variance
    variance = np.var(error)
    std_dev = np.sqrt(variance)
    
    # 2(d) & 2(e): DPCM and PSNR
    results_text = []
    results_text.append(f"Error Variance: {variance:.2f}")
    results_text.append(f"Error Std Dev: {std_dev:.2f}\n")
    
    for levels in [2, 4, 8]:
        unit_thresh, unit_recon = get_lloyd_max_table_unit_variance_laplacian(levels)
        
        # Scale by standard deviation
        thresholds = unit_thresh * std_dev
        reconstruction = unit_recon * std_dev
        
        results_text.append(f"--- Quantizer: {levels} levels ---")
        results_text.append(f"Thresholds: {np.round(thresholds, 2)}")
        results_text.append(f"Reconstruction levels: {np.round(reconstruction, 2)}")
        
        # Apply DPCM
        recon_img_array = dpcm(img_array, thresholds, reconstruction)
        
        # Calculate PSNR
        psnr = calculate_psnr(img_array, recon_img_array)
        results_text.append(f"PSNR: {psnr:.2f} dB\n")
        
        # Save image
        out_img = Image.fromarray(recon_img_array.astype(np.uint8))
        out_img.save(f'dpcm_reconstructed_{levels}_levels.png')
        
    with open('part_2_dpcm_results.txt', 'w') as f:
        f.write("\n".join(results_text))
        
if __name__ == "__main__":
    main()
