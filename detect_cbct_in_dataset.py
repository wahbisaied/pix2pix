"""
CBCT Detection in CT Dataset using Hounsfield Unit Analysis
Analyzes all NIfTI files to identify potential CBCT data based on HU values
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd

def analyze_hu_values(nifti_path):
    """Analyze Hounsfield Unit characteristics of a NIfTI file"""
    try:
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        # Remove zero/background pixels
        non_zero_data = data[data != 0]
        
        if len(non_zero_data) == 0:
            return None
        
        # Calculate HU statistics
        stats_dict = {
            'min_hu': float(np.min(non_zero_data)),
            'max_hu': float(np.max(non_zero_data)),
            'mean_hu': float(np.mean(non_zero_data)),
            'std_hu': float(np.std(non_zero_data)),
            'range_hu': float(np.max(non_zero_data) - np.min(non_zero_data)),
        }
        
        # Get voxel spacing
        voxel_spacing = nifti_img.header.get_zooms()
        stats_dict['voxel_x'] = float(voxel_spacing[0])
        stats_dict['voxel_y'] = float(voxel_spacing[1])
        stats_dict['voxel_z'] = float(voxel_spacing[2])
        
        # Check if isotropic (CBCT characteristic)
        stats_dict['is_isotropic'] = (abs(voxel_spacing[0] - voxel_spacing[1]) < 0.1 and 
                                     abs(voxel_spacing[1] - voxel_spacing[2]) < 0.1)
        
        return stats_dict
        
    except Exception as e:
        print(f"Error analyzing {nifti_path}: {e}")
        return None

def classify_imaging_modality(stats):
    """Classify as CT or CBCT based on HU characteristics"""
    
    cbct_score = 0
    reasons = []
    
    # HU Range Analysis
    if stats['min_hu'] > -500:
        cbct_score += 2
        reasons.append(f"Min HU too high: {stats['min_hu']:.1f}")
    
    if stats['max_hu'] < 2000:
        cbct_score += 1
        reasons.append(f"Max HU too low: {stats['max_hu']:.1f}")
    
    if stats['range_hu'] < 1500:
        cbct_score += 2
        reasons.append(f"HU range small: {stats['range_hu']:.1f}")
    
    # Voxel Spacing
    if stats['is_isotropic']:
        cbct_score += 3
        reasons.append(f"Isotropic voxels")
    
    # Mean HU Analysis
    if -200 < stats['mean_hu'] < 200:
        cbct_score += 1
        reasons.append(f"Mean HU suspicious: {stats['mean_hu']:.1f}")
    
    # Standard Deviation
    if stats['std_hu'] < 300:
        cbct_score += 1
        reasons.append(f"Low contrast: {stats['std_hu']:.1f}")
    
    # Classification
    if cbct_score >= 5:
        classification = "LIKELY_CBCT"
    elif cbct_score >= 3:
        classification = "POSSIBLE_CBCT"
    else:
        classification = "LIKELY_CT"
    
    return classification, cbct_score, reasons

def analyze_dataset(dataset_path):
    """Analyze entire dataset for CBCT detection"""
    
    print("="*60)
    print("CBCT DETECTION ANALYSIS")
    print("="*60)
    
    results = []
    
    for subset in ['trainA', 'trainB']:
        subset_path = os.path.join(dataset_path, subset)
        
        if not os.path.exists(subset_path):
            print(f"Warning: {subset_path} not found!")
            continue
        
        print(f"\nAnalyzing {subset}...")
        
        nifti_files = [f for f in os.listdir(subset_path) if f.endswith('.nii.gz')]
        print(f"Found {len(nifti_files)} NIfTI files")
        
        for i, filename in enumerate(nifti_files):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(nifti_files)}: {filename}")
            
            file_path = os.path.join(subset_path, filename)
            stats = analyze_hu_values(file_path)
            
            if stats is not None:
                classification, score, reasons = classify_imaging_modality(stats)
                
                result = {
                    'subset': subset,
                    'filename': filename,
                    'classification': classification,
                    'cbct_score': score,
                    'reasons': '; '.join(reasons),
                    **stats
                }
                results.append(result)
    
    return results

def generate_report(results, output_dir):
    """Generate CBCT detection report"""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'cbct_detection_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate text report
    report_path = os.path.join(output_dir, 'cbct_detection_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CBCT DETECTION REPORT\n")
        f.write("="*40 + "\n\n")
        
        # Summary
        total_files = len(df)
        likely_cbct = len(df[df['classification'] == 'LIKELY_CBCT'])
        possible_cbct = len(df[df['classification'] == 'POSSIBLE_CBCT'])
        likely_ct = len(df[df['classification'] == 'LIKELY_CT'])
        
        f.write(f"SUMMARY:\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Likely CBCT: {likely_cbct} ({likely_cbct/total_files*100:.1f}%)\n")
        f.write(f"Possible CBCT: {possible_cbct} ({possible_cbct/total_files*100:.1f}%)\n")
        f.write(f"Likely CT: {likely_ct} ({likely_ct/total_files*100:.1f}%)\n\n")
        
        # CBCT candidates
        cbct_candidates = df[df['classification'].isin(['LIKELY_CBCT', 'POSSIBLE_CBCT'])]
        
        if len(cbct_candidates) > 0:
            f.write("CBCT CANDIDATES:\n")
            f.write("-" * 20 + "\n")
            
            for _, row in cbct_candidates.iterrows():
                f.write(f"\nFile: {row['filename']}\n")
                f.write(f"Classification: {row['classification']}\n")
                f.write(f"Score: {row['cbct_score']}/10\n")
                f.write(f"HU Range: {row['min_hu']:.1f} to {row['max_hu']:.1f}\n")
                f.write(f"Mean HU: {row['mean_hu']:.1f}\n")
                f.write(f"Isotropic: {row['is_isotropic']}\n")
                f.write(f"Reasons: {row['reasons']}\n")
                f.write("-" * 30 + "\n")
    
    print(f"Report saved to: {report_path}")
    
    # Print results
    print(f"\n" + "="*40)
    print("RESULTS:")
    print(f"="*40)
    print(f"Total files: {total_files}")
    print(f"Likely CBCT: {likely_cbct}")
    print(f"Possible CBCT: {possible_cbct}")
    print(f"Likely CT: {likely_ct}")
    
    if likely_cbct > 0 or possible_cbct > 0:
        print(f"\n⚠️  CBCT DETECTED!")
        for _, row in cbct_candidates.iterrows():
            print(f"  - {row['filename']} ({row['classification']})")
    else:
        print(f"\n✅ All files appear to be CT data.")

def main():
    """Main function"""
    dataset_path = "D:\\ct_phases_datasets\\ct_phase0_dataset"
    output_dir = "C:\\Users\\Wahbi Saied\\Documents\\GitHub\\pix2pix"
    
    print("Starting CBCT detection analysis...")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Analyze dataset
    results = analyze_dataset(dataset_path)
    
    if results:
        generate_report(results, output_dir)
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()