#!/usr/bin/env python3
"""
Fast Data Loader - Load once, save processed data, then analyze interactively
"""

import sys
import os
import numpy as np
import h5py
import pickle
from pathlib import Path

# Add ARPES module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ARPES

def fast_load_h5(file_path, save_processed=True, output_dir="processed_data"):
    """
    Fast loading: only extract essential data, save as .npz for quick access
    """
    print("Fast loading ARPES data...")
    print(f"File: {file_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load with your function but optimize
    exit_slit_data, exit_tilt_data, hv_data, image_data, image_attrs = \
        ARPES.read_h5_file_with_h5py(file_path)
    
    # Extract scales from attributes (much faster than full file exploration)
    scales = extract_scales_fast(image_attrs, image_data.shape)
    
    # Get Fermi slice immediately
    intensity_fermi = get_fermi_slice_fast(image_data, image_attrs)
    
    # Create the data_array format your functions expect
    data_array = create_data_array_fast(intensity_fermi, scales)
    
    # Metadata dictionary
    metadata = {
        'file_path': file_path,
        'image_shape': image_data.shape,
        'intensity_shape': intensity_fermi.shape,
        'scales': scales,
        'image_attrs': image_attrs,
        'data_array_shape': data_array.shape,
        'timestamp': np.datetime64('now')
    }
    
    # Save everything
    if save_processed:
        # Save as compressed numpy file (much faster than HDF5 for analysis)
        npz_filename = output_path / f"{Path(file_path).stem}_processed.npz"
        np.savez_compressed(
            npz_filename,
            data_array=data_array,
            metadata=metadata,
            allow_pickle=True
        )
        print(f"✅ Saved processed data to: {npz_filename}")
        
        # Also save a small pickle for quick metadata access
        pickle_filename = output_path / f"{Path(file_path).stem}_metadata.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Saved metadata to: {pickle_filename}")
    
    print(f"Data ready! Shape: {data_array.shape}")
    print(f"Memory usage: ~{data_array.nbytes / 1e6:.1f} MB")
    
    return data_array, metadata

def extract_scales_fast(attrs, image_shape):
    """Extract scales directly from attributes - no file recursion"""
    scales = {}
    
    # Energy scale (Axis0)
    if 'Axis0.Scale' in attrs:
        start, step = attrs['Axis0.Scale']
        n_energies = image_shape[0] if len(image_shape) == 3 else 1
        energies = start + np.arange(n_energies) * step
        scales['energies'] = energies
    
    # Slit scale (Axis1 - kx)
    if 'Axis1.Scale' in attrs:
        start, step = attrs['Axis1.Scale']
        n_slit = image_shape[-1]
        slit = start + np.arange(n_slit) * step
        scales['slit'] = slit  # degrees
    else:
        scales['slit'] = np.arange(image_shape[-1])
    
    # Tilt scale (Axis2 - ky)
    if 'Axis2.Scale' in attrs:
        start, step = attrs['Axis2.Scale']
        n_tilt = image_shape[-2]
        tilt = start + np.arange(n_tilt) * step
        scales['tilt'] = tilt  # degrees
    else:
        scales['tilt'] = np.arange(image_shape[-2])
    
    # Physical parameters
    scales['hv'] = attrs.get('Excitation Energy (eV)', 25.0)
    scales['work_function'] = attrs.get('Work Function (eV)', 4.5)
    
    return scales

def get_fermi_slice_fast(image_data, attrs):
    """Get Fermi intensity slice quickly"""
    if len(image_data.shape) == 3:
        if 'Axis0.Scale' in attrs:
            # Kinetic energy scale
            start, step = attrs['Axis0.Scale']
            n_energies = image_data.shape[0]
            energies = start + np.arange(n_energies) * step
            # Fermi at highest kinetic energy
            fermi_idx = np.argmax(energies)
        else:
            fermi_idx = image_data.shape[0] - 1  # Last frame
        
        intensity = image_data[fermi_idx, :, :]
        print(f"  Fermi slice: index {fermi_idx}, KE = {energies[fermi_idx] if 'energies' in locals() else 'unknown'} eV")
    else:
        intensity = image_data
    
    return intensity

def create_data_array_fast(intensity, scales):
    """Create the (ny, nx, 3) data_array format"""
    ny, nx = intensity.shape
    kx_grid, ky_grid = np.meshgrid(scales['slit'], scales['tilt'])
    
    data_array = np.zeros((ny, nx, 3))
    data_array[:, :, 0] = ky_grid  # ky (tilt angles)
    data_array[:, :, 1] = kx_grid  # kx (slit angles)
    data_array[:, :, 2] = intensity  # intensity
    
    return data_array

def load_processed_data(npz_file):
    """Load previously saved processed data (super fast!)"""
    print(f"Loading processed data from: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    data_array = data['data_array']
    metadata = data['metadata'].item()
    print(f"✅ Loaded! Shape: {data_array.shape}")
    return data_array, metadata

if __name__ == "__main__":
    # Your data file
    file_path = r"D:\Siham\Data\FS_AN_25eV_LHQ_PHI_2p5_0000.h5"
    
    # Load once and save
    data_array, metadata = fast_load_h5(file_path, save_processed=True)
    
    # Quick preview
    print("\nQuick preview:")
    print(f"kx range: {metadata['scales']['slit'].min():.2f}° to {metadata['scales']['slit'].max():.2f}°")
    print(f"ky range: {metadata['scales']['tilt'].min():.2f}° to {metadata['scales']['tilt'].max():.2f}°")
    print(f"Max intensity: {data_array[:, :, 2].max():.1f}")