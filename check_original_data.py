import os
import glob

def check_original_data():
    """Check original data sources"""
    
    avg_src_dir = r"C:\Users\Wahbi Saied\OneDrive\avg"
    phases_src_dir = r"C:\Users\Wahbi Saied\OneDrive\final_data"
    
    print("🔍 Vérification des données originales...")
    
    # Check avg directory
    if os.path.exists(avg_src_dir):
        avg_folders = [f for f in os.listdir(avg_src_dir) if os.path.isdir(os.path.join(avg_src_dir, f))]
        print(f"📁 AVG directory: {len(avg_folders)} scan folders")
    else:
        print(f"❌ AVG directory not found: {avg_src_dir}")
        return
    
    # Check phases directory  
    if os.path.exists(phases_src_dir):
        phases_folders = [f for f in os.listdir(phases_src_dir) if os.path.isdir(os.path.join(phases_src_dir, f))]
        print(f"📁 Phases directory: {len(phases_folders)} scan folders")
    else:
        print(f"❌ Phases directory not found: {phases_src_dir}")
        return
    
    # Find matching pairs
    matching_pairs = []
    phase_0_available = []
    
    for scan_id in avg_folders:
        avg_scan_folder = os.path.join(avg_src_dir, scan_id)
        phases_scan_folder = os.path.join(phases_src_dir, scan_id)
        
        if os.path.isdir(phases_scan_folder):
            # Check for AVG file
            avg_files = glob.glob(os.path.join(avg_scan_folder, '*.nii*'))
            
            # Check for Phase 0 file
            phase_0_pattern = os.path.join(phases_scan_folder, '*Gated 0.0*.nii*')
            phase_0_files = glob.glob(phase_0_pattern)
            
            if avg_files and phase_0_files:
                matching_pairs.append(scan_id)
                phase_0_available.append(scan_id)
                print(f"✅ {scan_id}: AVG + Phase 0")
            elif avg_files:
                print(f"⚠️  {scan_id}: AVG only (no Phase 0)")
            else:
                print(f"❌ {scan_id}: No AVG file")
    
    print(f"\n📊 Résumé:")
    print(f"Total scan folders (AVG): {len(avg_folders)}")
    print(f"Total scan folders (Phases): {len(phases_folders)}")
    print(f"Matching pairs with Phase 0: {len(matching_pairs)}")
    
    # Check all phases availability
    print(f"\n🔍 Vérification de toutes les phases...")
    
    phases_count = {i: 0 for i in range(10)}
    
    for scan_id in matching_pairs[:5]:  # Check first 5 scans
        phases_scan_folder = os.path.join(phases_src_dir, scan_id)
        print(f"\n📁 {scan_id}:")
        
        for phase in range(10):
            phase_pattern = os.path.join(phases_scan_folder, f'*Gated {phase/10:.1f}*.nii*')
            phase_files = glob.glob(phase_pattern)
            
            if phase_files:
                phases_count[phase] += 1
                print(f"  ✅ Phase {phase}: {os.path.basename(phase_files[0])}")
            else:
                print(f"  ❌ Phase {phase}: Not found")
    
    print(f"\n📊 Phases disponibles (sur {len(matching_pairs)} scans):")
    for phase, count in phases_count.items():
        print(f"Phase {phase}: {count} scans")
    
    return matching_pairs

if __name__ == "__main__":
    check_original_data()