import subprocess
import time
import sys

def test_batch_size(batch_size, num_threads):
    """Test training speed with different batch sizes"""
    
    cmd = [
        sys.executable, "train.py",
        "--dataroot", "D:\\ct_phases_datasets\\ct_phase0_dataset",
        "--name", f"benchmark_bs{batch_size}",
        "--model", "pix2pix",
        "--dataset_mode", "robust_nifti",
        "--skip_corrupted",
        "--preprocess", "none",
        "--input_nc", "1",
        "--output_nc", "1",
        "--axial_slice",
        "--norm", "instance",
        "--netG", "unet_512",
        "--batch_size", str(batch_size),
        "--lambda_L1", "50",
        "--lr", "0.0002",
        "--resize_to", "512",
        "--display_freq", "1000",  # Less frequent display
        "--save_epoch_freq", "1000",  # Don't save during benchmark
        "--num_threads", str(num_threads),
        "--n_epochs", "1",  # Only 1 epoch for benchmark
        "--n_epochs_decay", "0"
    ]
    
    print(f"🧪 Testing batch_size={batch_size}, num_threads={num_threads}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ Batch {batch_size}: {duration:.1f} seconds")
            return duration
        else:
            print(f"❌ Batch {batch_size}: Failed - {result.stderr[:100]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ Batch {batch_size}: Timeout (>10 min)")
        return None
    except Exception as e:
        print(f"❌ Batch {batch_size}: Error - {e}")
        return None

def benchmark_all():
    """Benchmark different configurations"""
    
    print("🚀 Benchmark des batch sizes sur RTX 5090")
    print("⏱️  Test de 1 époque pour chaque configuration")
    print()
    
    configs = [
        (16, 16),   # Original
        (24, 24),   # Medium
        (32, 32),   # Current
        (48, 32),   # High batch, same threads
        (64, 32),   # Very high batch
    ]
    
    results = {}
    
    for batch_size, num_threads in configs:
        duration = test_batch_size(batch_size, num_threads)
        if duration:
            results[(batch_size, num_threads)] = duration
        print()
    
    print("📊 Résultats du benchmark:")
    print("Batch Size | Threads | Time (s) | Speed")
    print("-----------|---------|----------|-------")
    
    if results:
        best_time = min(results.values())
        for (bs, nt), duration in sorted(results.items()):
            speedup = best_time / duration
            print(f"    {bs:2d}     |   {nt:2d}    |  {duration:6.1f}  | {speedup:.2f}x")
        
        best_config = min(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Meilleure config: batch_size={best_config[0][0]}, threads={best_config[0][1]}")
        print(f"⏱️  Temps: {best_config[1]:.1f}s")

if __name__ == "__main__":
    benchmark_all()