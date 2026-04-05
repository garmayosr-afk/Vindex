import cv2
import numpy as np

def test_fft(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = mag.shape
    r = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) > r ** 2
    
    high_freq = mag[mask]
    
    var = np.var(high_freq)
    mean = np.mean(high_freq)
    std = np.std(high_freq)
    
    # Peak counting: spikes that are anomalously high
    peaks_3std = np.sum(high_freq > (mean + 3 * std))
    peaks_4std = np.sum(high_freq > (mean + 4 * std))
    peaks_5std = np.sum(high_freq > (mean + 5 * std))
    
    ratio_high_to_low = np.mean(mag[mask]) / (np.mean(mag[~mask]) + 1e-8)
    
    print(f"--- {image_path} ---")
    print(f"Var: {var:.2f}, ratio: {ratio_high_to_low:.4f}")
    print(f"Peaks (>3std): {peaks_3std}")
    print(f"Peaks (>4std): {peaks_4std}")
    print(f"Peaks (>5std): {peaks_5std}")
    print()

test_fft("test_images/real.jpg")
test_fft("fraud_images/photoshopped.jpg")
test_fft("fraud_images/ai generated.png")
