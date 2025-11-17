@echo off
echo ========================================
echo Testing Single Patient - 50 Slices
echo ========================================

python test.py ^
    --dataroot D:\ct_phases_datasets\ct_phase0_dataset ^
    --name ct_phase0_generator_5090 ^
    --model pix2pix ^
    --direction AtoB ^
    --epoch latest ^
    --dataset_mode robust_nifti ^
    --input_nc 1 ^
    --output_nc 1 ^
    --netG unet_512 ^
    --norm instance ^
    --preprocess none ^
    --axial_slice ^
    --num_test 50

echo.
echo Test completed! Check results in: results\ct_phase0_generator_5090\test_latest\
pause