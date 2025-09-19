import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit, minimize
import os
import re
import h5py
import glob

# ~~~~~~~~~~~~~~~ TABLE OF CONTENTS ~~~~~~~~~~~~~~~~ #
#       1.      READ AND WRITE DATA FILES       l. 34
#       2.      ARPES DATA TREATMENT            l. 138
#       3.      ARPES TOOLS                     l. 223
#       4.      EQUATION FIT                    l. 361
#       5.      EDC / MDC                       l. 424
#       6.      PLOT FUNCTIONS                  l. 516
#       7.      CASSIOPEE                       l. 612
#       8.      OTHER VERY USEFULL FUNCTIONS    l. 849
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#from config import get_material
#material = get_material()
#a, b, c = material.lattice

#______________________________________________________#

############## READ AND WRITE DATA FILES ###############
#______________________________________________________#

def read_h5_file_with_h5py(file_path): 
    with h5py.File(file_path, 'r') as h5_file:
        exit_slit_data = None
        exit_tilt_data = None
        hv_data = None
        image_data = None
        image_dset = None
        image_attrs = None  # <-- pour stocker les attributs

        def process_group(group, prefix=''):
            nonlocal exit_slit_data, hv_data, image_data, image_dset, exit_tilt_data, image_attrs

            for key in group.keys():
                print(key)
                item = group[key]
                path = f"{prefix}/{key}" if prefix else key

                if isinstance(item, h5py.Dataset):
                    data = item[:]
                    if "Image Data" in path:
                        image_data = []
                        image_dset = item
                        image_attrs = dict(item.attrs)  # <-- récupère les attributs
                        print(f"Found image data: {path}")
                        print("Attributes:", image_attrs)

                        ndim = data.ndim
                        if ndim == 2: 
                            print("only one acquisition stack")
                            image_data = data[np.newaxis, :, :]   # shape = (1, Ny, Nx)
                        else:
                            n_frame = data.shape[2]
                            print(n_frame, "acquisitions stack")
                            image_data = np.transpose(data, (2, 0, 1))  # shape = (Nframes, Ny, Nx)

                    elif "Axis0.Scale" in path:
                        hv_data = data
                    elif "Axis1.Scale" in path:
                        exit_slit_data = data
                    elif "Axis2.Scale" in path:
                        exit_tilt_data = data

                elif isinstance(item, h5py.Group):
                    process_group(item, prefix=path)

        process_group(h5_file)

        return exit_slit_data, exit_tilt_data, hv_data, image_data, image_attrs
    
def write_h5py_from_dataset(dataset, filename,energies, slit,theta ): # impove this to inplement theta, energy and slit 
    with h5py.File('./'+filename+'.h5', "w") as f:
        dset = f.create_dataset("dataset", data=dataset)
        dset.attrs["Energy"] = np.array([energies[0], find_step_from_list(energies)])
        dset.attrs["Slit"] = np.array([slit[0], find_step_from_list(slit)])
        dset.attrs["Theta_scan"] = np.array([theta[0], find_step_from_list(theta)])

def open_data(file_list_dir): #format .txt
    data=[]
    for i,name in enumerate(file_list_dir):
        file_panda = pd.read_csv(name,delimiter='\t')
        data.append(file_panda.to_numpy()) 
    return data

def open_FS(file_dir, para_slit, para_angle): # not so usefull
    '''open data_array 1 energy file'''
    '''para_slit = (start value, step value) (same for angle)'''
    # Lire les données
    file_panda = pd.read_csv(file_dir, delimiter='\t', header=None)
    data_I = file_panda.to_numpy()  # Intensité (2D)
    ny, nx = data_I.shape  # rows = Y (tilt), cols = X (slit)

    # Créer les coordonnées en X (slit) et Y (tilt)
    slit = para_slit[0] + np.arange(nx) * para_slit[1]
    angle = para_angle[0] + np.arange(ny) * para_angle[1]
    slit = np.array(slit)
    angle = np.array(angle)

    kx_grid, ky_grid = np.meshgrid(slit, angle)  # kx = colonnes (X), ky = lignes (Y)
    data_array = np.zeros((ny, nx, 3))
    data_array[:, :, 0] = ky_grid  # Y
    data_array[:, :, 1] = kx_grid  # X
    data_array[:, :, 2] = data_I  # Intensité

    return data_array

def para2list(para,nx): 
    '''para = (initial_value, step_value)
       nx = len_list'''
    list0 = para[0] + np.arange(nx) * para[1]
    return list0

def name2energy(folder_path): # this function is not important 
    params = []
    filenames = []

    for filename in os.listdir(folder_path):
        match = re.search(r"FS_ARPES_(\d+)p(\d+)", filename)
        if match:
            integer_part = match.group(1)
            decimal_part = match.group(2)
            param = float(f"{integer_part}.{decimal_part}")
            params.append(param)
            filenames.append(filename)
    
    return filenames, params

#______________________________________________________#

################# ARPES DATA TREATMENT #################
#______________________________________________________#

def rotation(coordinates, angle):  # angle in degrees
    ''' correction of phi angle'''
    angle_rad = np.radians(angle)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad),  np.cos(angle_rad)]])

    if coordinates.ndim == 2 and coordinates.shape[1] == 2:
        # Format [N,2] : liste simple de points
        return coordinates @ rot_matrix.T

    elif coordinates.ndim == 3 and coordinates.shape[2] == 2:
        # Format [ny,nx,2] : meshgrid
        flat_coords = coordinates.reshape(-1, 2)
        rotated = flat_coords @ rot_matrix.T
        return rotated.reshape(coordinates.shape)

    else:
        raise ValueError("Input should be shape (N,2) or (ny,nx,2)")

def shift(data, x_shift, y_shift):
    ''' correction of tilt and theta '''
    shift_vec = np.array([x_shift, y_shift])

    if data.ndim == 2 and data.shape[1] == 2:
        return data + shift_vec

    elif data.ndim == 3 and data.shape[2] == 2:
        return data + shift_vec.reshape(1, 1, 2)

    else:
        raise ValueError("Input should be shape (N,2) or (ny,nx,2)")

def angle_to_momentum(data, Ec, unit): 
    """
    Convertit des angles (en degrés) en impulsions (k-space), en fonction de l'énergie cinétique E.
    
    data : ndarray de shape (N,2) ou (ny,nx,2)
    Ec : kinetic energy (en eV)
    unit : 'BZ' or 'Angstrom-1'
    a : unit cell (Å)
    """
    
    hbar = 1.054e-34  # J.s
    m_e = 9.1e-31     # kg
    eV = 1.6e-19      # J
    pi = np.pi

    if unit == 'BZ':
        const = np.sqrt(2 * m_e * Ec * eV) / hbar * 1e-10 * a / pi * 2  # BZ units
    elif unit == 'Angstrom-1':
        const = np.sqrt(2 * m_e * Ec * eV) / hbar * 1e-10               # Å⁻¹
    else:
        raise ValueError("unit must be 'BZ' or 'Angstrom-1'")

    data_conv = np.copy(data)

    if data.ndim == 2 and data.shape[1] == 2:
        # Format [N, 2]
        data_conv[:, 0] = const * np.sin(np.radians(data[:, 0]))
        data_conv[:, 1] = const * np.sin(np.radians(data[:, 1]))

    elif data.ndim == 3 and data.shape[2] == 2:
        # Format [ny, nx, 2]
        data_conv[:, :, 0] = const * np.sin(np.radians(data[:, :, 0]))
        data_conv[:, :, 1] = const * np.sin(np.radians(data[:, :, 1]))

    else:
        raise ValueError("data must be shape (N,2) or (ny,nx,2)")

    return data_conv

def operation(data,angle,shiftx,shifty,E,unit, sym=False): 
    ''' do all what we need !'''
    data = shift(data,shiftx,shifty)
    data = rotation(data,angle)
    data = angle_to_momentum(data,E, unit)
    if sym == True:
        data = symmetrization(data,True,True,True)
    return data

#______________________________________________________#

##################### ARPES TOOLS ######################
#______________________________________________________#

def symmetrization(data, V, H, D):
    ''' symmetrizes a list of points (kx,ky)'''
    sym_data = [data]  # Start with the original data

    if V:
        data_V = data.copy()
        data_V[:, 0] = -data[:, 0]
        sym_data.append(data_V)

    if H:
        data_H = data.copy()
        data_H[:, 1] = -data[:, 1]
        sym_data.append(data_H)

    if D:
        data_D = data.copy()
        data_D1 = data.copy()
        data_D2 = data.copy()
        data_D3 = data.copy()
        data_D4 = data.copy()
        data_D[:, 0] = -data[:, 1]
        data_D[:, 1] = -data[:, 0]
        data_D1[:, 0] = data[:, 1]
        data_D1[:, 1] = data[:, 0]    
        data_D2[:, 0] = -data[:, 0]
        data_D2[:, 1] = -data[:, 1]
        data_D3[:, 0] = -data[:, 1]
        data_D3[:, 1] = data[:, 0]    
        data_D4[:, 0] = data[:, 1]
        data_D4[:, 1] = -data[:, 0]   
        sym_data.append(data_D2)
        sym_data.append(data_D1)        
        sym_data.append(data_D)
        sym_data.append(data_D3)
        sym_data.append(data_D4)

    return np.concatenate(sym_data)

def multi_cut_parameters(data,theta_list,bounds= ([0.01,0.6,0.0,0.0],[0.15,0.8,10,1])):
    weigth = []
    width = []
    pos = []
    for i, theta in enumerate(theta_list):
        cut, proj = cut_MDC(data,theta,(1,-1),radius = 0.008, N_points=700,length=1)
        points = group_close_points(cut,proj,1e-3)
        fit, mdc = MDC_fit((points[:,3], points[:,2]),lorentzien1,bounds)
        weigth.append(mdc[2])
        width.append(mdc[0])
        pos.append(mdc[1])
    return theta_list, weigth, width, pos

def find_endpoint(origin, theta_deg, length):
    """
    Calcule le point final d’un segment partant de `origin` dans la direction `theta_deg`
    (entre 0° et 90° selon le quadrant) avec une longueur maximale fixée (par défaut : 1).
    
    theta_deg : angle local par rapport à l’axe x du coin d’origine (entre 0 et 90)
    origin : tuple (x0, y0)
    length : longueur maximale du segment
    """
    x0, y0 = origin
    theta_deg = float(theta_deg)

    if not (0 <= theta_deg <= 90):
        raise ValueError("theta_deg doit être entre 0 et 90 degrés")

    # Convertit l'angle selon le quadrant (même logique que précédemment)
    if x0 < 0 and y0 > 0:         # en haut à gauche → vers bas droite
        theta = -theta_deg
    elif x0 > 0 and y0 > 0:       # en haut à droite → vers bas gauche
        theta = 180 + theta_deg
    elif x0 > 0 and y0 < 0:       # en bas à droite → vers haut gauche
        theta = 180 - theta_deg
    elif x0 < 0 and y0 < 0:       # en bas à gauche → vers haut droite
        theta = theta_deg
    else:
        raise ValueError("L’origine doit être dans un des quatre coins")

    theta_rad = np.deg2rad(theta)
    dx, dy = np.cos(theta_rad), np.sin(theta_rad)

    # Point final à distance "length" dans la direction (dx, dy)
    x_end = x0 + length * dx
    y_end = y0 + length * dy

    return (x_end, y_end)

def TB_fit(data,p0,bounds,equation,E_0,unit):
    lim, label = set_unit(unit)

    def error_function(p0,kx, ky, E_observed,unit):
        E_predicted = equation(kx, ky,*p0,  unit)
        return np.sum((E_predicted - E_observed)**2)
    
    kx_ = data[:,0]
    ky_ = data[:,1]
    kx_data = np.array([x/lim for x in kx_[0:len(kx_)-2]])
    ky_data = np.array([x/lim for x in ky_[0:len(kx_)-2]])
    E_observed_data = np.array([E_0]*(len(kx_)-2))

    # Minimiser l'erreur pour trouver les meilleurs paramètres
    result = minimize(error_function, p0, args=(kx_data, ky_data, E_observed_data, lim),bounds=bounds, method='Nelder-Mead')

    # Paramètres ajustés
    optimized_params = result.x
    tolerance = 1e-3

    # Définir une grille de valeurs pour kx et ky
    kx_values = np.linspace(-1,1, 1000)
    ky_values = np.linspace(-1,1, 1000)
    KX, KY = np.meshgrid(kx_values, ky_values)

    # Calculer les valeurs d'énergie pour chaque point de la grille

    E_values = equation(KX, KY, *optimized_params, 1)

    # Trouver les points où l'énergie est proche de E_0
    close_points = np.abs(E_values - E_0) <  tolerance
    # Extraire les valeurs de kx et ky correspondantes
    kx_solutions = KX[close_points]
    ky_solutions = KY[close_points]

    data_fit = np.zeros((len(kx_solutions),2))
    data_fit[:,0] = kx_solutions
    data_fit[:,1] = ky_solutions

    return data_fit,optimized_params

#___________________________________________#

############## EQUATION FIT #################
#___________________________________________#

def d_wave_gap(angle, gap,shift):
    return -gap*(np.cos(2*angle*np.pi/180))+shift

def polynom(x,a,b,c):
    return(a*x**2+b*x+c)

def gaussian(x,a,b,c,bgd):
    return c*np.exp(-(x-b)**2/2/a**2)/a+bgd

def E_k(kx, ky,mu, t0, t1, t2, t3, tbi, tz, unit):
    term00 = - t0* 2 * (np.cos(kx/unit*np.pi) + np.cos(ky/unit*np.pi))       
    term01 = -  t0*4 * t1 * np.cos(kx / unit*np.pi) * np.cos(ky / unit*np.pi)
    term02 = - t0* 2 * t2 * (np.cos(2 * kx / unit*np.pi) + np.cos(2 * ky / unit*np.pi))
    term03 = -  t0*4 * t3 * (np.cos(2 * kx / unit*np.pi) * np.cos(ky / unit*np.pi) + np.cos(2 * ky / unit*np.pi) * np.cos(kx / unit*np.pi))
    pot = mu
    return term00+term01+term02 +pot +term03

def E_k_bound(kx, ky,mu, t0, t1, t2, t3, tbi, unit):
    term00 = - t0* 2 * (np.cos(kx/unit*np.pi) + np.cos(ky/unit*np.pi))       
    term01 = -  t0*4 * t1 * np.cos(kx / unit*np.pi) * np.cos(ky / unit*np.pi)
    term02 = - t0* 2 * t2 * (np.cos(2 * kx / unit*np.pi) + np.cos(2 * ky / unit*np.pi))
    term03 = -  t0*4 * t3 * (np.cos(2 * kx / unit*np.pi) * np.cos(ky / unit*np.pi) + np.cos(2 * ky / unit*np.pi) * np.cos(kx / unit*np.pi))
    termbi = -2*tbi*t0*(np.cos(kx / unit*np.pi)-np.cos(ky / unit*np.pi))**2 
    pot = mu
    return term00+term01+term02 +pot +term03+termbi

def E_k_antibound(kx, ky,mu, t0, t1, t2, t3, tbi, unit):
    term00 = - t0* 2 * (np.cos(kx/unit*np.pi) + np.cos(ky/unit*np.pi))       
    term01 = -  t0*4 * t1 * np.cos(kx / unit*np.pi) * np.cos(ky / unit*np.pi)
    term02 = - t0* 2 * t2 * (np.cos(2 * kx / unit*np.pi) + np.cos(2 * ky / unit*np.pi))
    term03 = -  t0*4 * t3 * (np.cos(2 * kx / unit*np.pi) * np.cos(ky / unit*np.pi) + np.cos(2 * ky / unit*np.pi) * np.cos(kx / unit*np.pi))
    termbi = -2*tbi*t0*(np.cos(kx / unit*np.pi)-np.cos(ky / unit*np.pi))**2 

    pot = mu
    return term00+term01+term02 +pot +term03-termbi

def E_kF(kx, mu, t0, t1, t2, t3, unit): # kx=ky
    cos1 = np.cos(kx / unit*np.pi)
    cos2 = np.cos(2 * kx / unit*np.pi)
    
    term00 = -t0 * 4 * cos1
    term01 = -t0 * 4 * t1 * cos1 * cos1
    term02 = -t0 * 4 * t2 * cos2
    term03 = -t0 * 8 * t3 * cos2 * cos1
    return term00 + term01 + term02 + term03 + mu

def lorentzien3(x,A,B,C,alp,bet,gam,A1,B1,C1,bgd):
    return  (C*A/np.pi/(A**2+(x-B)**2)+C1*A1/np.pi/(A1**2+(x-B1)**2)+gam*alp/np.pi/(alp**2+(x-bet)**2))+bgd

def lorentzien2(x,A,B,C,alp,bet,gam,bgd):
    return  (C*A/np.pi/(A**2+(x-B)**2)+gam*alp/np.pi/(alp**2+(x-bet)**2))+bgd

def lorentzien21(x,A,B,C,bet,bgd):
    return  (C*A/np.pi/(A**2+(x-B)**2)+C*A/np.pi/(A**2+(x-bet)**2))+bgd

def lorentzien1(x,A,B,C,bgd):
    return  (C*A/np.pi/(A**2+(x-B)**2))+bgd

#___________________________________________#

################ EDC / MDC ##################
#___________________________________________#

def cut_MDC(data,angle,origin, radius = 0.1,N_points = 500,length=1):
    end = find_endpoint(origin,angle,length)
    kx_values = np.linspace(origin[0], end[0], N_points)
    ky_values = np.linspace(origin[1], end[1], N_points)
    segment = np.vstack((kx_values, ky_values)).T
    kx_data = data[:, :, 0].flatten()
    ky_data = data[:, :, 1].flatten()
    Intensity = data[:,:,2].flatten()
    cut = []
    indices = []
    for px, py in segment:
        dist = np.sqrt((kx_data - px)**2 + (ky_data - py)**2)
        indices.append(np.where(dist <= radius)[0])
    indices = np.concatenate(indices)
    cut = np.column_stack((kx_data[indices],ky_data[indices],Intensity[indices]))
        # vecteur du segment
    segment_vec = np.array(end) - np.array(origin)
    segment_len = np.linalg.norm(segment_vec)
    segment_dir = segment_vec / segment_len  # vecteur unitaire

    # vecteur des points par rapport à l'origine
    points_vec = cut[:, :2] - origin

    # projection scalaire des points sur le segment (distance le long du segment)
    proj_length = np.dot(points_vec, segment_dir)

    # On peut aussi filtrer les points hors segment (optionnel)
    mask = (proj_length >= 0) & (proj_length <= segment_len)
    cut_filtered = cut[mask]
    proj_length_filtered = proj_length[mask]

    # Sort points as the projection
    sort_idx = np.argsort(proj_length_filtered)
    cut_sorted = cut_filtered[sort_idx]
    proj_length_sorted = proj_length_filtered[sort_idx]
    
    return cut_sorted, proj_length_sorted

def MDC_fit(data,equation,bounds):
    datax,datay = data
    mdc, pcov = curve_fit(equation, datax,datay, bounds =bounds) # fonction, x, y 
    data_fit = np.zeros((500,2))
    data_fit[:,0] = np.linspace(datax[0], datax[-1],500)
    data_fit[:,1] = equation(data_fit[:,0],*mdc)
    return data_fit,mdc


def group_close_points(cut_data, k_proj, delta=1e-3):
    """
    cut_data : array Nx3 with kx, ky, intensity
    k_proj : array N of projeted datas
    delta : radius 
    """
    # Sorted by K_proj
    sorted_indices = np.argsort(k_proj)
    cut_sorted = cut_data[sorted_indices]
    k_sorted = k_proj[sorted_indices]

    grouped = []
    current_group = [cut_sorted[0]]
    current_k = [k_sorted[0]]

    for i in range(1, len(k_sorted)):
        if abs(k_sorted[i] - current_k[-1]) < delta:
            current_group.append(cut_sorted[i])
            current_k.append(k_sorted[i])
        else:
            group_arr = np.array(current_group)
            k_avg = np.mean(current_k)
            kx_avg = np.mean(group_arr[:, 0])
            ky_avg = np.mean(group_arr[:, 1])
            I_avg  = np.mean(group_arr[:, 2])
            grouped.append([kx_avg, ky_avg, I_avg, k_avg])
            current_group = [cut_sorted[i]]
            current_k = [k_sorted[i]]

    # Add the last group
    if current_group:
        group_arr = np.array(current_group)
        k_avg = np.mean(current_k)
        kx_avg = np.mean(group_arr[:, 0])
        ky_avg = np.mean(group_arr[:, 1])
        I_avg  = np.mean(group_arr[:, 2])
        grouped.append([kx_avg, ky_avg, I_avg, k_avg])

    return np.array(grouped)

#________________________________________________#

################ PLOT FUNCTIONS ##################
#________________________________________________#

def plot_energy_integrated_map(dataset, energies, E_center, dE=0.01, cmap="terrain_r"):
    """
    plot an integrated map in energy around E_center with width 2*dE
    
    dataset : np.array (nTheta, nE, nAngle)
    energies : np.array (nE,)
    E_center : float -> Middle energy
    dE : float -> integration half width
    """
    # indices des énergies dans l'intervalle
    mask = (energies >= (E_center - dE)) & (energies <= (E_center + dE))
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise ValueError(f"No energy found in this range [{E_center-dE:.2f}, {E_center+dE:.2f}] eV")

    # moyenne sur ces énergies
    I_mean = dataset[idx, :, :].mean(axis=1)  # shape = (nTheta, nAngle)

    # tracer
    plt.figure(figsize=(6,5))
    plt.imshow(I_mean.T, 
               origin="lower",
               aspect="auto",
               interpolation="bilinear",
               cmap=cmap,
               extent=[0, dataset.shape[0], -15,+15])
    plt.colorbar(label="Intensity (a.u.)")
    plt.title(f"Integrated map {E_center:.2f} eV (±{dE} eV)")
    plt.xlabel(r"$\theta$ (°)")
    plt.ylabel("Slit (°)")
    plt.show()

def stack_EDC(list_edc, E_list, thetas=0, Ef=0, offset=1, c= "#D17E02"):
    n_curves = len(list_edc)
    fig, ax = plt.subplots(figsize=(6,5.5))

    for i in range(n_curves):
        data_offset = list_edc[ i] + i * offset # add offset to each curve 
        plt.plot(E_list, data_offset,  color =c, lw=0.8)
        if thetas != 0:
            plt.text(E_list[0], data_offset[10]+0.3, fr"$\theta$ = {thetas[i]}°", fontsize=9, va='center', ha='left', color='black')

    # Ligne de Fermi
    plt.axvline(Ef, ymin=0, ymax=1, ls="--", color="black", label="Fermi level")

    # Axes et légende
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (offset)")
    plt.legend()
    plt.tight_layout()
    return fig, ax

def plot_ARPES_FS(data,E=0, unit=None, c = 'terrain_r'): # plot ARPES data_array <= cut at 1 energy of the dataset
    lim, label = set_unit(unit)
    fig, ax = plt.subplots(figsize=(6,5.5))
    ax.set_xlabel("kx" + (f" ({unit})" if unit else ""))
    ax.set_ylabel("ky" + (f" ({unit})" if unit else ""))
    E_str = str(E)
    ax.set_title(r"ARPES $1^{st}$ Brillouin Zone ($E_b=$"+E_str+"$~meV$)")
    for val in [-lim, 0, lim]:
        ax.vlines(val, ymin=-lim, ymax=lim, linestyles='--', color='black')
        ax.hlines(val, xmin=-lim, xmax=lim, linestyles='--',color='black')
    X = data[:, :, 0]
    Y = data[:, :, 1]
    Z = data[:, :, 2]

    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap = c)
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    return fig, ax

def plot_data_FS(data,unit,s): # create a BZ base and scatter list of (kx,ky)
    lim, label = set_unit(unit)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], s=s)
    for val in [-lim, 0, lim]:
        ax.vlines(val, ymin=-lim, ymax=lim, linestyles='--', color='black')
        ax.hlines(val, xmin=-lim, xmax=lim, linestyles='--',color='black')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel(r'$k_x$ '+label)
    ax.set_ylabel(r'$k_y$ '+label)
    return fig, ax

def plot_sub_function(data, mdc, main_function, sub_functions, s=3, ax=None): # if fit with several funcions => can plot pretty subfunctions (written for MDC cuts)
    '''sub_functions = ((equation1, parameters of eq1), (equation2,parameters of eq2))'''
    datax, datay = data
    color_list = ["sandybrown", "lightcoral"] # Have to add more colors in this list !!

    # Utiliser l'axe fourni, sinon en créer un
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure  # utile si tu veux return le fig

    ax.scatter(datax, datay, color="saddlebrown", s=s)
    ax.plot(datax, datay, color="saddlebrown", linewidth=1)
    
    bgd = mdc[-1]
    x_fit = np.linspace(datax[0], datax[-1], 500)
    y_bgd = [bgd for _ in range(len(x_fit))]
    y_fit_main = main_function(x_fit, *mdc)

    if sub_functions[0] != main_function:
        for f, (function, popt) in enumerate(sub_functions):
            y_fit = function(x_fit, *popt)
            ax.fill_between(x_fit, y_fit, bgd, color=color_list[f], alpha=0.3)
        ax.plot(x_fit, y_fit_main, color='red', linewidth=1)
    else:
        ax.fill_between(x_fit, y_fit_main, bgd, color='red', alpha=0.3)

    ax.fill_between(x_fit, y_bgd, color="grey", hatch="///", alpha=0.3)
    ax.grid(True)
    ax.set_xlabel("$k$ centered in M")
    ax.set_ylabel("Intensity (a.u.)")

    return fig, ax

#________________________________________________#

################### CASSIOPEE ####################
#________________________________________________#

def read_FS_txt(filename):
    """
    Read file QFS/FS .txt and return (energies, angles, I)
    If the file does not contain an acceptable matrix, I will be np.array([])
    """
    with open(filename, "r", encoding="latin1") as f:
        lines = f.readlines()

    energies = None
    angles = None
    raw = []

    for line in lines:
        clean = line.strip()

        # récupérer les scales si présentes (ligne entière après '=' contient beaucoup de valeurs)
        if clean.lower().startswith("dimension 1 scale"):
            # prendre tout ce qui suit '=' et split proprement
            try:
                values = re.split(r"\s+", clean.split("=", 1)[1].strip())
                energies = np.array(list(map(float, values)))
            except Exception:
                energies = None
        elif clean.lower().startswith("dimension 2 scale"):
            try:
                values = re.split(r"\s+", clean.split("=", 1)[1].strip())
                angles = np.array(list(map(float, values)))
            except Exception:
                angles = None
        else:
            # détecter les lignes numériques (énergie + intensités) : ligne avec au moins 2 nombres
            parts = re.split(r"\s+", clean)
            if len(parts) >= 2:
                try:
                    nums = list(map(float, parts))
                    raw.append(nums)
                except ValueError:
                    # ligne non numérique -> ignorer
                    pass

    raw = np.array(raw)
    if raw.size == 0:
        return energies, angles, np.array([])   # fichier sans matrice

    # si la première colonne est l'énergie (très probable), alors intensités = raw[:,1:]
    if raw.ndim == 2 and angles is not None and raw.shape[1] == len(angles) + 1:
        I = raw[:, 1:]
    else:
        # fallback : si shape concorde sans colonne énergie, on retourne raw
        # (par sécurité on essaie d'inférer)
        I = raw
    return energies, angles, I

def numbers_tuple_from_name(fname):
    """Return tule of integrer found in the filename, ex: 'FS1_100.txt' -> (1,100)."""
    base = os.path.basename(fname)
    nums = re.findall(r"\d+", base)
    if not nums:
        return ()
    return tuple(int(n) for n in nums)

def load_FS_dataset(folder):
    """
    Charge tous les fichiers spectres du dossier en s'assurant d'un tri numérique.
    - ignore les fichiers se terminant par '_i.txt' (fichiers d'info)
    - ignore les fichiers qui n'ont pas de matrice (I.size == 0)
    - n'empile que les fichiers dont la forme correspond à la référence (premier fichier valide)
    Retour : energies, angles, dataset (nScan, nE, nAngle), files_used (triés)
    """
    # lister tous les .txt
    all_files = sorted(glob.glob(os.path.join(folder, "*.txt")))

    # étape 1 : lire TOUS les fichiers et collecter les entrées utiles
    entries = []
    for f in all_files:
        name = os.path.basename(f)
        # ignorer explicitement les fichiers d'info nommés '*_i.txt' (ou contenant '_i' juste avant '.txt')
        if re.search(r'_i\.txt$', name, flags=re.IGNORECASE):
            # print(f"Ignoring info file: {name}")
            continue

        E, A, I = read_FS_txt(f)
        if I.size == 0:
            # pas de matrice -> ignorer
            # print(f"Ignored (no matrix): {name}")
            continue

        nums = numbers_tuple_from_name(name)
        entries.append({'file': f, 'nums': nums, 'E': E, 'A': A, 'I': I})

    if len(entries) == 0:
        raise RuntimeError("No data file detected")

    # étape 2 : déterminer la forme de référence (prend la plus fréquente ou la première)
    shapes = [e['I'].shape for e in entries]
    # choisir la shape la plus fréquente comme référence
    from collections import Counter
    shape_counts = Counter(shapes)
    ref_shape, _ = shape_counts.most_common(1)[0]

    # filtrer les entries ayant la bonne shape
    filtered = []
    skipped = []
    for e in entries:
        if e['I'].shape == ref_shape:
            filtered.append(e)
        else:
            skipped.append((os.path.basename(e['file']), e['I'].shape))

    if len(filtered) == 0:
        raise RuntimeError(f"No entry with reference shape {ref_shape}.")

    # étape 3 : trier numériquement selon la tuple 'nums'
    # Si certains noms n'ont pas de nums, on place leur key en fin (empty tuple sorts before non-empty, so use special key)
    def sort_key(entry):
        # retourner une clé comparable : (has_nums:0/1, nums tuple)
        nums = entry['nums']
        if nums:
            return (0, nums)   # 0 = a des nombres -> avant
        else:
            return (1, ())     # 1 = pas de nombres -> après

    filtered_sorted = sorted(filtered, key=sort_key)

    # étape 4 : empiler dataset dans l'ordre trié
    dataset = np.stack([e['I'] for e in filtered_sorted], axis=2)
    # energies & angles (on prend celles du premier trié)
    E0 = filtered_sorted[0]['E']
    A0 = filtered_sorted[0]['A']
    files_used = [os.path.basename(e['file']) for e in filtered_sorted]

    # rapporter fichiers ignorés pour debug
    if skipped:
        print("⚠️ Fichiers ignorés (different shape) :")
        for name, shape in skipped:
            print("   -", name, "shape =", shape)

    return np.array(E0), np.array(A0), dataset, files_used
import os
import re

def extract_thetas_from_folder(folder):
    """
    Parcourt tous les fichiers *_i.txt dans un dossier
    et extrait les valeurs de theta (deg).
    
    Args:
        folder (str): chemin vers le dossier
    
    Returns:
        list[float]: liste des valeurs de theta triées par index de fichier
    """
    
    def extract_num(filename):
        """Récupère le numéro du fichier pour trier correctement"""
        match = re.search(r"(\d+)_i\.txt$", filename)
        return int(match.group(1)) if match else -1
    
    thetas = []
    files = [f for f in os.listdir(folder) if f.endswith("_i.txt")]
    files = sorted(files, key=extract_num)  # tri numérique
    
    for fname in files:
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                if "theta (°)" in line:
                    val = float(line.split(":")[1].strip())
                    thetas.append(val)
                    break
    
    return thetas

def check_file_order(folder):
    """Print numeric order before importing"""

    all_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    candidates = []
    for f in all_files:
        name = os.path.basename(f)
        if re.search(r'_i\.txt$', name, flags=re.IGNORECASE):
            continue
        nums = numbers_tuple_from_name(name)
        candidates.append((nums, name))
    # trier et afficher
    candidates_sorted = sorted(candidates, key=lambda x: (0, x[0]) if x[0] else (1, ()))
    print("Detected order (nums, filename) :")
    for nums, name in candidates_sorted:
        print(nums, name)


def read_FS_txt(filename):
    with open(filename, "r", encoding="latin1") as f:
        lines = f.readlines()

    energies, angles = [], []
    raw = []

    for line in lines:
        # Récupérer l’échelle des énergies
        if "Dimension 1 scale" in line:
            try:
                energies = list(map(float, line.split("=")[1].split()))
            except:
                pass

        # Récupérer l’échelle des angles
        elif "Dimension 2 scale" in line:
            try:
                angles = list(map(float, line.split("=")[1].split()))
            except:
                pass

        # Détecter les vraies lignes de données (floats)
        else:
            parts = line.strip().split()
            if len(parts) > 1:
                try:
                    values = list(map(float, parts))
                    raw.append(values)
                except ValueError:
                    pass

    raw = np.array(raw)

    # Vérifier la consistance
    if raw.ndim == 2 and len(angles) > 0 and raw.shape[1] == len(angles) + 1:
        I = raw[:, 1:]  # enlève la colonne énergie
    else:
        I = np.array([])  # fichier sans données utiles

    return np.array(energies), np.array(angles), I

#________________________________________________#

######### OTHER VERY USEFULL FUNCTIONS ###########
#________________________________________________#

def find_index_from_value(list0,value):
    index = min(range(len(list0)), key=lambda i: abs(list0[i] - value))    
    return index

def find_step_from_list(list):
    return (list[-1]-list[0])/len(list)
    
def set_unit(unit):
    if unit == 'BZ':
        lim = 1
        label = r"($\pi/a$)"
    elif unit == 'Angstrom-1':
        lim = np.pi/a/2
        label  = r"($Å^{-1}$)"
    return (lim, label)

#________________________________________________#