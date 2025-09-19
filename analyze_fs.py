#!/usr/bin/env python3
"""
Fermi Surface Analysis Script
Uses ARPES.py as a module to load and visualize .h5 data
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path to import ARPES.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your ARPES module
import ARPES

def main():
    # File path to your data
    file_path = r"D:\Siham\Data\FS_AN_25eV_LHQ_PHI_2p5_0000.h5"
    
    print("Loading ARPES data...")
    print("="*50)
    
    # Load the data using your function
    exit_slit_data, exit_tilt_data, hv_data, image_data, image_attrs = \
        ARPES.read_h5_file_with_h5py(file_path)
    
    # Print info about the loaded data
    print_data_info(exit_slit_data, exit_tilt_data, hv_data, image_data, image_attrs)
    
    # Process the data for Fermi surface
    data_fs = process_fermi_surface(image_data, exit_slit_data, exit_tilt_data, 
                                   hv_data, image_attrs, file_path)
    
    if data_fs is not None:
        # Plot the Fermi surface
        plot_fermi_surface(data_fs)
    else:
        print("Failed to process data for Fermi surface plot.")

def print_data_info(exit_slit, exit_tilt, hv, image, attrs):
    """Print useful information about the loaded data"""
    print(f"Exit slit data: {exit_slit.shape if exit_slit is not None else 'None'}")
    print(f"Exit tilt data: {exit_tilt.shape if exit_tilt is not None else 'None'}")
    print(f"HV data (energy?): {hv.shape if hv is not None else 'None'}")
    print(f"Image data shape: {image.shape}")
    print(f"Attributes keys: {list(attrs.keys()) if attrs else 'None'}")
    
    # Print some attribute values that might be useful
    useful_attrs = ['Axis0.Scale', 'Axis1.Scale', 'Axis2.Scale', 
                   'Axis1.FirstVal', 'Axis1.StepVal', 'Axis2.FirstVal', 'Axis2.StepVal',
                   'Excitation Energy (eV)', 'Work Function (eV)']
    for attr in useful_attrs:
        if attr in attrs:
            print(f"  {attr}: {attrs[attr]}")

def process_fermi_surface(image_data, exit_slit_data, exit_tilt_data, hv_data, attrs, file_path):
    """
    Process the ARPES data to create a Fermi surface map
    Returns data_array (ny, nx, 3) with [ky, kx, intensity]
    """
    try:
        # Get angle scales
        slit, tilt = get_angle_scales(exit_slit_data, exit_tilt_data, attrs, image_data.shape)
        print(f"Angle scales - Slit: {slit.shape}, Tilt: {tilt.shape}")
        print(f"Slit range: {slit.min():.2f}° to {slit.max():.2f}°")
        print(f"Tilt range: {tilt.min():.2f}° to {tilt.max():.2f}°")
        
        # Get intensity data (slice at Fermi)
        intensity = get_fermi_intensity(image_data, hv_data, attrs)
        print(f"Intensity shape: {intensity.shape}")
        
        # Create data_array like in your open_FS function
        ny, nx = intensity.shape
        kx_grid, ky_grid = np.meshgrid(slit, tilt)
        
        data_array = np.zeros((ny, nx, 3))
        data_array[:, :, 0] = ky_grid  # ky (tilt angles)
        data_array[:, :, 1] = kx_grid  # kx (slit angles) 
        data_array[:, :, 2] = intensity  # intensity
        
        # Convert to momentum space
        data_fs = convert_to_kspace(data_array, attrs, file_path)
        
        return data_fs
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_angle_scales(exit_slit, exit_tilt, attrs, image_shape):
    """Extract angle scales from data or attributes"""
    
    # Try to use provided scales first
    if exit_slit is not None and len(exit_slit) == image_shape[-1]:
        slit = exit_slit
    else:
        # Try attributes
        slit = get_scale_from_attrs(attrs, 'Axis1', image_shape[-1])
        if slit is None:
            # Default: pixel indices (you'll need to calibrate)
            print("⚠️  Warning: Using pixel indices for slit (calibrate this!)")
            slit = np.arange(image_shape[-1])
    
    if exit_tilt is not None and len(exit_tilt) == image_shape[-2]:
        tilt = exit_tilt
    else:
        # Try attributes
        tilt = get_scale_from_attrs(attrs, 'Axis2', image_shape[-2])
        if tilt is None:
            print("⚠️  Warning: Using pixel indices for tilt (calibrate this!)")
            tilt = np.arange(image_shape[-2])
    
    return slit, tilt

def get_scale_from_attrs(attrs, axis_name, n_points):
    """Try to reconstruct scale from HDF5 attributes (handles [start, step] format)"""
    scale_key = f'{axis_name}.Scale'
    if scale_key in attrs:
        scale_data = attrs[scale_key]
        if len(scale_data) == 2:  # [start, step] format
            start, step = scale_data
            scale = start + np.arange(n_points) * step
            return scale
        elif len(scale_data) == n_points:  # Full scale list
            return scale_data
    # Also check old FirstVal/StepVal (for compatibility)
    first_key = f'{axis_name}.FirstVal'
    step_key = f'{axis_name}.StepVal'
    if first_key in attrs and step_key in attrs:
        start = attrs[first_key]
        step = attrs[step_key]
        scale = start + np.arange(n_points) * step
        return scale
    return None

def get_fermi_intensity(image_data, hv_data, attrs):
    """Get intensity at Fermi level (max kinetic energy)"""
    if image_data.ndim == 3:
        # 3D data: (n_frames/energy, ny, nx)
        if 'Axis0.Scale' in attrs:
            # Reconstruct energy scale from attributes [start, step]
            energy_start, energy_step = attrs['Axis0.Scale']
            n_energies = image_data.shape[0]
            energies = energy_start + np.arange(n_energies) * energy_step
            # Fermi at highest kinetic energy
            fermi_idx = np.argmax(energies)
            print(f"Slicing at Fermi index {fermi_idx} (KE={energies[fermi_idx]:.3f} eV)")
        else:
            # No energy scale, take last frame (highest energy)
            fermi_idx = image_data.shape[0] - 1
            print(f"No energy scale found, using last frame {fermi_idx} (assumed Fermi)")
        
        intensity = image_data[fermi_idx, :, :]
    else:
        # 2D data: already at Fermi
        intensity = image_data[0, :, :] if image_data.ndim == 3 else image_data
    
    return intensity

def convert_to_kspace(data_array, attrs, file_path):
    """Convert angle data to momentum space using your operation function"""
    
    # Get kinetic energy from attributes (Excitation Energy is photon energy ~25 eV)
    # Fermi KE = hv - work function, but since we slice at max KE, use max KE for conversion
    if 'Excitation Energy (eV)' in attrs:
        hv = attrs['Excitation Energy (eV)']
        work_function = attrs.get('Work Function (eV)', 0.0)
        Ec = hv - work_function  # Theoretical max KE at Fermi
        print(f"Max KE at Fermi: {Ec:.2f} eV (hv={hv:.2f} eV, WF={work_function:.2f} eV)")
    else:
        Ec = 25.0  # Fallback from filename
        print(f"Using kinetic energy: {Ec} eV (no attrs found)")
    
    # Calibration parameters (you'll need to adjust these)
    angle_correction = 0.0   # degrees (phi rotation)
    shiftx = 0.0             # slit (kx) shift in degrees  
    shifty = 0.0             # tilt (ky) shift in degrees
    unit = 'Angstrom-1'      # or 'BZ' if you define lattice constant below
    
    # For 'BZ' units, define lattice constant a (in Angstroms) here, e.g.:
    # a = 3.8  # Example for some cuprate; set your material's value
    # Then change unit = 'BZ' above
    
    print(f"Converting to k-space with:")
    print(f"  Ec = {Ec} eV, unit = {unit}")
    print(f"  Corrections: angle={angle_correction}°, shift=({shiftx}, {shifty})°")
    
    # Apply your operation function
    try:
        # Get just the coordinates part (ny, nx, 2)
        coords = data_array[:, :, :2]
        k_coords = ARPES.operation(coords, angle_correction, shiftx, shifty, Ec, unit, sym=False)
        
        # Rebuild full data array with k-space coordinates
        data_fs = np.zeros_like(data_array)
        data_fs[:, :, 0] = k_coords[:, :, 0]  # ky
        data_fs[:, :, 1] = k_coords[:, :, 1]  # kx
        data_fs[:, :, 2] = data_array[:, :, 2]  # intensity
        
        return data_fs
        
    except Exception as e:
        print(f"Error in k-space conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_energy_from_filename(filepath):
    """Extract photon/kinetic energy from filename"""
    import re
    basename = os.path.basename(filepath)
    # Look for patterns like "25eV", "25EV", "25", etc.
    match = re.search(r'(\d+(?:\.\d+)?)[eE]?[vV]?', basename)
    if match:
        return float(match.group(1))
    return None

def plot_fermi_surface(data_fs, unit='Angstrom-1'):
    """Plot the Fermi surface using your plotting function"""
    print("Generating Fermi surface plot...")
    
    # Temporary fix for set_unit error: define 'a' globally if using 'BZ'
    # (Uncomment if you switch to unit='BZ' and set your value)
    # a = 3.8  # Your lattice constant in Å
    # globals()['a'] = a  # Make 'a' available to ARPES.set_unit
    
    # Use your existing plot function
    fig, ax = ARPES.plot_ARPES_FS(data_fs, E=0, unit=unit, c='terrain_r')
    
    # Add some extra formatting
    plt.suptitle('Fermi Surface Map', fontsize=14, y=0.95)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    print("Plot displayed! Close the window to continue.")
    
    return fig, ax

if __name__ == "__main__":
    main()