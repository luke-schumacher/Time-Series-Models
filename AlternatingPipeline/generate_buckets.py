"""
Script to generate all buckets using trained models.
"""
import os
import torch
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_SAVE_DIR, BUCKET_SIZE
from models.exchange_model import create_exchange_model
from models.examination_model import create_examination_model
from generation.bucket_generator import BucketGenerator

def main():
    print("Initializing Bucket Generation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Exchange Model
    print("\nLoading Exchange Model...")
    exchange_model = create_exchange_model()
    exchange_path = os.path.join(MODEL_SAVE_DIR, 'exchange', 'exchange_model_final.pt')
    if os.path.exists(exchange_path):
        exchange_model.load_state_dict(torch.load(exchange_path, map_location=device))
        print(f"  Loaded weights from {exchange_path}")
    else:
        print(f"  Warning: Exchange model weights not found at {exchange_path}")
        return

    # 2. Load Examination Model
    print("\nLoading Examination Model...")
    examination_model = create_examination_model()
    examination_path = os.path.join(MODEL_SAVE_DIR, 'examination', 'examination_model_final.pt')
    if os.path.exists(examination_path):
        examination_model.load_state_dict(torch.load(examination_path, map_location=device))
        print(f"  Loaded weights from {examination_path}")
    else:
        print(f"  Warning: Examination model weights not found at {examination_path}")
        return

    # 3. Initialize Bucket Generator
    print("\nInitializing Bucket Generator...")
    generator = BucketGenerator(
        exchange_model=exchange_model,
        examination_model=examination_model,
        device=device
    )

    # 4. Generate Buckets
    # Using a smaller bucket size for testing if needed, or stick to BUCKET_SIZE (1000)
    # The user asked for "help with generating the buckets", so 1000 is likely expected.
    print(f"\nGenerating all buckets ({BUCKET_SIZE} samples each)...")
    print("This may take a while depending on your hardware.")
    
    generator.generate_all_buckets(num_samples=BUCKET_SIZE)

    # 5. Save Buckets
    print("\nSaving buckets to disk...")
    generator.save_buckets()
    
    print("\nBucket generation complete!")

if __name__ == "__main__":
    main()
