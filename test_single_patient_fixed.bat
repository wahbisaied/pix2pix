@echo off
echo ========================================
echo Testing Single Patient - Fixed Naming
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
    --num_test 50 ^
    --results_dir ./results/single_patient_test

echo.
echo Test completed! Check results in: results\single_patient_test\
echo Each slice now has proper patient_name_slice### naming!
pause