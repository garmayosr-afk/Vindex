import os
import glob
import json
from PIL import Image
import imagehash
import argparse

def precompute_phash(directory, output_file="fraud_db.json"):
    """
    Scans a directory for images and computes their perceptual hashes.
    Stores the results in a JSON file mapping filenames to hashes.
    """
    fraud_db = {}
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
        # uppercase extensions too
        image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
        
    print(f"Found {len(image_paths)} images in {directory}. Processing...")
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                # Calculate perceptual hash
                hash_val = str(imagehash.phash(img))
                filename = os.path.basename(img_path)
                fraud_db[filename] = hash_val
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            
    with open(output_file, 'w') as f:
        json.dump(fraud_db, f, indent=4)
        
    print(f"Successfully saved {len(fraud_db)} hashes to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute pHash for fraud database")
    parser.add_argument("directory", help="Directory containing fraudulent images")
    parser.add_argument("-o", "--output", default="fraud_db.json", help="Output JSON file path")
    args = parser.parse_args()
    
    precompute_phash(args.directory, args.output)
