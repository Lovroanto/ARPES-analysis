#!/usr/bin/env python3
"""
Interactive ARPES Analysis - Load pre-processed data and experiment freely
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add ARPES module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ARPES

# Load your pre-processed data (super fast!)
processed_dir = "processed_data"
npz_file = Path(processed_dir) / "FS_AN_25eV_LHQ_PHI_2p5_0000_processed.npz"

if not npz_file.exists():
    print(f"❌ Processed file not found: {npz_file}")
    print("Run load_data.py first!")
    sys.exit(1)

print("🚀 Loading pre-processed data...")
data_array, metadata = load_processed_data(npz_file)
scales = metadata['scales']

print(f"Data loaded in milliseconds! Shape: {data_array.shape}")
print(f"kx range: {scales['slit'].min():.1f}° → {scales['slit'].max():.1f}°")
print(f"ky range: {scales['tilt'].min():.1f}° → {scales['tilt'].max():.1f}°")

# Now you can experiment freely - change this section as much as you want!
def analyze_and_plot():
    """Your analysis playground - modify this function freely!"""
    
    # Parameters to tweak (no reloading needed!)
    Ec = scales['hv'] - scales['work_function']  # ~20.5 eV
    unit = 'Angstrom-1'
    angle_correction = 0.0  # degrees
    shiftx = 0.0  # slit shift
    shifty = 0.0  # tilt shift
    
    print(f"\nAnalyzing with:")
    print(f"  Ec = {Ec:.1f} eV, unit = {unit}")
    print(f"  Corrections: angle={angle_correction}°, shift=({shiftx}, {shifty})°")
    
    # Convert to k-space (uses your ARPES.operation function)
    coords = data_array[:, :, :2]
    k_coords = ARPES.operation(coords, angle_correction, shiftx, shifty, Ec, unit, sym=False)
    
    # Rebuild k-space data
    data_fs = np.zeros_like(data_array)
    data_fs[:, :, 0] = k_coords[:, :, 0]  # ky
    data_fs[:, :, 1] = k_coords[:, :, 1]  # kx
    data_fs[:, :, 2] = data_array[:, :, 2]  # intensity
    
    # Plot (uses your ARPES.plot_ARPES_FS function)
    fig, ax = ARPES.plot_ARPES_FS(data_fs, E=0, unit=unit, c='terrain_r')
    plt.suptitle('Interactive Fermi Surface', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Return data for further analysis
    return data_fs

# Run the analysis
if __name__ == "__main__":
    data_fs = analyze_and_plot()
    
    # Keep Python running for interactive mode
    print("\n🎯 Interactive mode - try:")
    print("  - Modify parameters in analyze_and_plot() and run again")
    print("  - Use data_fs for further analysis (MDC, EDC, etc.)")
    print("  - Add your own plotting functions here")
    
    # Optional: enter interactive Python shell
    try:
        import code
        code.interact(local=locals())
    except:
        input("\nPress Enter to exit...")