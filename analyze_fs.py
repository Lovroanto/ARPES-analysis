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
                   'Axis1.FirstVal', 'Axis1.StepVal', 'Axis2.FirstVal', 'Axis2.StepVal']
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
        
        # Get intensity data (slice at Fermi or use full if 2D)
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
    """Try to reconstruct scale from HDF5 attributes"""
    if f'{axis_name}.FirstVal' in attrs and f'{axis_name}.StepVal' in attrs:
        first_val = attrs[f'{axis_name}.FirstVal']
        step_val = attrs[f'{axis_name}.StepVal']
        scale = first_val + np.arange(n_points) * step_val
        return scale
    elif f'{axis_name}.Scale' in attrs:
        scale = attrs[f'{axis_name}.Scale']
        if len(scale) == n_points:
            return scale
    return None

def get_fermi_intensity(image_data, hv_data, attrs):
    """Get intensity at Fermi level or integrate around it"""
    if image_data.ndim == 3:
        # 3D data: (n_frames/energy, ny, nx)
        if hv_data is not None:
            # Find Fermi energy (assuming E=0 is Fermi for binding energy)
            fermi_idx = np.argmin(np.abs(hv_data))
            print(f"Slicing at Fermi index {fermi_idx} (E={hv_data[fermi_idx]:.3f} eV)")
            intensity = image_data[fermi_idx, :, :]
        else:
            # No energy scale, take middle or first frame
            fermi_idx = image_data.shape[0] // 2
            print(f"No energy scale found, using frame {fermi_idx}")
            intensity = image_data[fermi_idx, :, :]
    else:
        # 2D data: already at Fermi
        intensity = image_data[0, :, :] if image_data.ndim == 3 else image_data
    
    return intensity

def convert_to_kspace(data_array, attrs, file_path):
    """Convert angle data to momentum space using your operation function"""
    
    # Get kinetic energy from filename or attributes
    Ec = extract_energy_from_filename(file_path)
    if Ec is None:
        # Try attributes or default
        if 'PhotonEnergy' in attrs:
            Ec = attrs['PhotonEnergy']
        else:
            Ec = 25.0  # from filename
        print(f"Using kinetic energy: {Ec} eV")
    
    # Calibration parameters (you'll need to adjust these)
    angle_correction = 0.0   # degrees (phi rotation)
    shiftx = 0.0             # slit shift (degrees)
    shifty = 0.0             # tilt shift (degrees)
    unit = 'Angstrom-1'      # or 'BZ' if you define lattice constant
    
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