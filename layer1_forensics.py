import os
import json
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageStat
import imagehash
import argparse
from PIL import ImageEnhance

def analyze_fft(image_path):
    """
    Analyzes the frequency spectrum of the image to detect AI generation artifacts.
    Returns: Float 0-1 (higher implies stronger AI characteristics)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Analyze high-frequency characteristics
    h, w = magnitude_spectrum.shape
    r = min(h, w) // 4
    # Create mask for high frequencies (block out the center low-freq circle)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) > r ** 2
    
    high_freq_spectrum = magnitude_spectrum[mask]
    
    # Calculate variance of high frequencies as a simple artifact metric
    # AI models often leave periodic high-freq "checkerboard" artifacts that cause spikes
    if len(high_freq_spectrum) == 0:
        return 0.0
        
    variance = np.var(high_freq_spectrum)
    
    # Normalize variance to a 0-1 score (dummy normalization thresholds for demonstration)
    variance_min = 100.0
    variance_max = 300.0
    
    score = (variance - variance_min) / (variance_max - variance_min)
    return float(np.clip(score, 0.0, 1.0))

def analyze_ela(image_path, quality_resave=90):
    """
    Error Level Analysis to detect Photoshop/editing modifications.
    Returns: Float 0-1 (higher implies higher likelihood of editing/splicing)
    """
    original = Image.open(image_path).convert('RGB')
    
    # Save temporarily to a lower quality JPEG to expose compression differences
    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality_resave)
    
    resaved = Image.open(temp_path)
    
    # Calculate difference
    diff = ImageChops.difference(original, resaved)
    
    # Enhance difference to see it clearer (optional, but good for scoring)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    enhancer = ImageEnhance.Brightness(diff)
    diff = enhancer.enhance(scale)
    
    # Clean up
    os.remove(temp_path)
    
    # Evaluate score:
    # A modified image will have areas of very high difference compared to the rest
    stat = ImageStat.Stat(diff)
    # Using the standard deviation of differences as a score
    # High standard deviation implies localized editing
    std_dev = max(stat.stddev)
    
    # Normalize (dummy thresholds)
    std_min = 20.0
    std_max = 80.0
    
    score = (std_dev - std_min) / (std_max - std_min)
    return float(np.clip(score, 0.0, 1.0))

def analyze_phash(image_path, db_path="fraud_db.json"):
    """
    Computes perceptual hash to detect duplicates in a database.
    Returns: Float 0-1 (1.0 = exact duplicate found, 0.0 = totally unique)
    """
    if not os.path.exists(db_path):
        # No DB, cannot find duplicates
        return 0.0
        
    with open(db_path, "r") as f:
        fraud_db = json.load(f)
        
    if not fraud_db:
        return 0.0
        
    img = Image.open(image_path)
    current_hash = imagehash.phash(img)
    
    min_distance = float('inf')
    
    for filename, hash_str in fraud_db.items():
        db_hash = imagehash.hex_to_hash(hash_str)
        distance = current_hash - db_hash
        if distance < min_distance:
            min_distance = distance
            
    # Normalize distance to score.
    # Distance of 0 = exact match (Score 1.0).
    # Distance > 15 is generally considered a different image
    max_threshold = 15.0
    if min_distance > max_threshold:
        return 0.0
        
    score = 1.0 - (min_distance / max_threshold)
    return float(np.clip(score, 0.0, 1.0))

def generate_report(image_path, db_path="fraud_db.json"):
    """
    Combines all Layer 1 forensics into a unified report and score.
    """
    fft_score = analyze_fft(image_path)
    ela_score = analyze_ela(image_path)
    phash_score = analyze_phash(image_path, db_path)
    
    # Weighted fusion of the micro-level scores
    # Example weights: pHash is conclusive if 1.0, ELA is very telling, FFT is indicative
    
    # If pHash is a strong match, that highly indicates a known fraud image
    if phash_score > 0.8:
        final_fraud_score = phash_score
    else:
        # Otherwise, weight the generative and editor artifacts
        final_fraud_score = (fft_score * 0.4) + (ela_score * 0.6)
        
    final_fraud_score = float(np.clip(final_fraud_score, 0.0, 1.0))
    
    report = {
        "verdict": "Synthetic/Edited" if final_fraud_score > 0.5 else "Likely Authentic",
        "fraud_score": round(final_fraud_score, 4),
        "details": {
            "fft_ai_artifact_score": round(fft_score, 4),
            "ela_tampering_score": round(ela_score, 4),
            "phash_duplicate_score": round(phash_score, 4)
        }
    }
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Layer 1 Forensics on an Image")
    parser.add_argument("image", help="Path to the query image")
    parser.add_argument("--db", default="fraud_db.json", help="Path to the fraud database JSON")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        exit(1)
        
    print(f"Analyzing {args.image}...")
    report = generate_report(args.image, args.db)
    
    print("\n--- FORENSIC REPORT ---")
    print(json.dumps(report, indent=4))
