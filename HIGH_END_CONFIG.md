# High-End System Configuration
## RTX 5090 32GB + Ryzen 9 9950X + 192GB RAM

### Optimized Training Command
```bash
python train.py --dataroot D:\ct_phases_dataset --name ct_phase0_generator_op --model pix2pix --dataset_mode robust_nifti --skip_corrupted --preprocess none --input_nc 1 --output_nc 1 --axial_slice --norm instance --netG unet_512 --batch_size 16 --lambda_L1 50 --lr 0.0002 --resize_to 512 --display_freq 50 --save_epoch_freq 5 --num_threads 16
```

### Key Changes for Your System:
- `--batch_size 16` (4x larger - much faster training)
- `--lr 0.0002` (slightly higher LR for larger batch)
- `--num_threads 16` (utilize your 16-core CPU)
- `--display_freq 50` (more frequent updates)

### Expected Performance:
- **Training Speed**: ~8-10 minutes per epoch (vs 36 min)
- **Memory Usage**: ~20-25GB VRAM
- **Total Training Time**: ~15-20 hours for 100 epochs

### If Memory Issues, Try:
```bash
--batch_size 12  # Still 3x faster than original
```

### Multi-GPU Ready (if you add another GPU):
```bash
torchrun --nproc_per_node=2 train.py [same parameters] --norm sync_instance --batch_size 32
```