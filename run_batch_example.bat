@echo off
REM Example batch file to run the patient processing
REM Modify the paths according to your setup

python batch_process_patients.py ^
    --data_dir "C:\Users\wahbi\Downloads\seg\data" ^
    --model_name "ct_phase0_generator_optimized" ^
    --epoch "50" ^
    --output_dir ".\best_results" ^
    --checkpoints_dir "C:\Users\wahbi\Downloads\seg\DG-20250924T214348Z-1-001\DG" ^
    --patient_range 1 6

pause