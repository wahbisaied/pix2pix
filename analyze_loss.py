import re
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_log(log_file_path):
    """Parse the loss log file and extract loss values"""
    
    epochs = []
    g_gan_losses = []
    g_l1_losses = []
    d_real_losses = []
    d_fake_losses = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match loss entries
    pattern = r'\[Rank 0\] \(epoch: (\d+), iters: \d+, time: [\d.]+, data: [\d.]+\) , G_GAN: ([\d.]+), G_L1: ([\d.]+), D_real: ([\d.]+), D_fake: ([\d.]+)'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch, g_gan, g_l1, d_real, d_fake = match
        epochs.append(int(epoch))
        g_gan_losses.append(float(g_gan))
        g_l1_losses.append(float(g_l1))
        d_real_losses.append(float(d_real))
        d_fake_losses.append(float(d_fake))
    
    return epochs, g_gan_losses, g_l1_losses, d_real_losses, d_fake_losses

def analyze_best_epoch(epochs, g_l1_losses):
    """Find the best epoch based on G_L1 loss (lower is better)"""
    
    # Group by epoch and calculate average G_L1 loss per epoch
    epoch_losses = {}
    for epoch, loss in zip(epochs, g_l1_losses):
        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append(loss)
    
    # Calculate average loss per epoch
    avg_losses = {}
    for epoch, losses in epoch_losses.items():
        avg_losses[epoch] = np.mean(losses)
    
    # Find best epochs
    sorted_epochs = sorted(avg_losses.items(), key=lambda x: x[1])
    
    return sorted_epochs, avg_losses

def plot_loss_trends(epochs, g_l1_losses, avg_losses):
    """Plot loss trends"""
    
    # Calculate moving average for smoother visualization
    unique_epochs = sorted(set(epochs))
    epoch_avg_losses = [avg_losses[epoch] for epoch in unique_epochs]
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual points (lighter)
    plt.scatter(epochs, g_l1_losses, alpha=0.3, s=1, color='lightblue', label='Individual G_L1 losses')
    
    # Plot epoch averages
    plt.plot(unique_epochs, epoch_avg_losses, 'r-', linewidth=2, label='Average G_L1 per epoch')
    
    # Find and mark the best epoch
    best_epoch = min(avg_losses.items(), key=lambda x: x[1])
    plt.axvline(x=best_epoch[0], color='green', linestyle='--', linewidth=2, 
                label=f'Best Epoch: {best_epoch[0]} (G_L1: {best_epoch[1]:.3f})')
    
    plt.xlabel('Epoch')
    plt.ylabel('G_L1 Loss')
    plt.title('Generator L1 Loss Over Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    log_file = r"C:\Users\Wahbi Saied\Documents\GitHub\pix2pix\checkpoints\ct_phase0_generator_5090\loss_log.txt"
    
    print("=" * 60)
    print("PIX2PIX TRAINING ANALYSIS - PHASE 0")
    print("=" * 60)
    
    # Parse the log file
    epochs, g_gan_losses, g_l1_losses, d_real_losses, d_fake_losses = parse_loss_log(log_file)
    
    if not epochs:
        print("❌ No loss data found in the log file!")
        return
    
    print(f"📊 Analyzed {len(epochs)} training iterations")
    print(f"📈 Training epochs: {min(epochs)} to {max(epochs)}")
    
    # Analyze best epochs
    sorted_epochs, avg_losses = analyze_best_epoch(epochs, g_l1_losses)
    
    print("\n" + "=" * 60)
    print("🏆 TOP 10 BEST EPOCHS (Based on G_L1 Loss - Lower is Better)")
    print("=" * 60)
    
    for i, (epoch, avg_loss) in enumerate(sorted_epochs[:10]):
        print(f"{i+1:2d}. Epoch {epoch:3d}: G_L1 = {avg_loss:.4f}")
    
    # Find the absolute best epoch
    best_epoch, best_loss = sorted_epochs[0]
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED MODEL TO USE:")
    print("=" * 60)
    print(f"✅ BEST EPOCH: {best_epoch}")
    print(f"✅ BEST G_L1 LOSS: {best_loss:.4f}")
    print(f"✅ MODEL FILE: {best_epoch}_net_G.pth")
    
    # Check if it's from the decay phase (typically after epoch 150)
    if best_epoch > 150:
        print("✅ This is from the DECAY PHASE - excellent!")
        print("   The model has converged well with learning rate decay.")
    else:
        print("ℹ️  This is from the MAIN TRAINING PHASE")
        print("   Consider also checking models from the decay phase (epoch 150+)")
    
    # Analysis of training phases
    print("\n" + "=" * 60)
    print("📈 TRAINING PHASE ANALYSIS:")
    print("=" * 60)
    
    main_phase_epochs = [e for e in avg_losses.keys() if e <= 150]
    decay_phase_epochs = [e for e in avg_losses.keys() if e > 150]
    
    if main_phase_epochs:
        main_avg = np.mean([avg_losses[e] for e in main_phase_epochs])
        print(f"Main Phase (1-150):   Avg G_L1 = {main_avg:.4f}")
    
    if decay_phase_epochs:
        decay_avg = np.mean([avg_losses[e] for e in decay_phase_epochs])
        print(f"Decay Phase (150+):   Avg G_L1 = {decay_avg:.4f}")
        
        if decay_avg < main_avg:
            print("✅ Decay phase shows BETTER performance - use decay phase model!")
        else:
            print("⚠️  Main phase shows better performance - unusual but possible")
    
    # Stability analysis
    print("\n" + "=" * 60)
    print("📊 TRAINING STABILITY:")
    print("=" * 60)
    
    recent_epochs = sorted_epochs[-20:]  # Last 20 epochs by performance
    recent_avg = np.mean([loss for _, loss in recent_epochs])
    
    print(f"Recent epochs average G_L1: {recent_avg:.4f}")
    
    if best_loss < recent_avg * 0.9:
        print("✅ Training shows good convergence - best model is significantly better")
    else:
        print("⚠️  Training might still be improving - consider training longer")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATION:")
    print("=" * 60)
    print(f"USE MODEL: {best_epoch}_net_G.pth")
    print(f"REASON: Lowest G_L1 loss ({best_loss:.4f}) indicates best image reconstruction")
    print("NEXT STEPS:")
    print("1. Copy this model for inference")
    print("2. Use same training settings for other phases")
    print("3. Monitor G_L1 loss as primary metric")
    
    # Plot the trends
    plot_loss_trends(epochs, g_l1_losses, avg_losses)

if __name__ == "__main__":
    main()