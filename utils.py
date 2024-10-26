import sanity
import visualize

import numpy as np

import os
from warnings import warn
import time
import json
import h5py

def spiral_to_undersampled(spiral_kspace_dict, R):
    """
    Undersamples the spiral kspace data with acceleration factor R.
    
    Args:
    spiral_kspace_dict (dict): Dict storing spiral k-space and trajectory data
    R (int): Acceleration factor.

    Returns:
    undersampled_spiral_kspace_dict (dict): Dictionary holding undersampled
        spiral k-space data, trajectory, and ISMRMRD header. keys are 'kspace',
        'traj', and 'metadata'
    """
    # Get traj and kspace
    traj = spiral_kspace_dict['traj']
    kspace = spiral_kspace_dict['kspace']

    # Get metadata
    metadata = spiral_kspace_dict['metadata']
    ninterleaves = metadata['spiralParameters']['ninterleaves']
    number_of_slices = kspace.shape[0]
    number_of_coils = sanity.num_coils_check(kspace, metadata)
    number_of_samples = int(traj.shape[1]/ninterleaves)
    
    # -------------------------------------------------------------------------
    # Reshape traj into 2 x Nk x Ni
    # Reshape kspace into Ns x Nc x Nk x Ni
    # -------------------------------------------------------------------------
    traj = np.reshape(traj, (2, number_of_samples, ninterleaves), order='F')
    kspace = np.reshape(kspace, (number_of_slices, number_of_coils, number_of_samples, ninterleaves), order='F')

    # -------------------------------------------------------------------------
    # Discard interleaves according to the acceleration factor
    # Select every R sample along the 4th dimension (the interleaf dimension)
    # -------------------------------------------------------------------------
    undersampled_traj = traj[:, :, ::R]
    undersampled_kspace = kspace[:, :, :, ::R]

    # -------------------------------------------------------------------------
    # Reshape them back into their original form
    # -------------------------------------------------------------------------
    # traj into 2 x (Nk * Ni_new)
    # kspace into Ns x Nc x (Nk * Ni_new)
    new_ninterleaves = undersampled_traj.shape[-1]
    undersampled_traj = np.reshape(undersampled_traj, 
                                   (2, number_of_samples * new_ninterleaves),
                                   order='F')
    undersampled_kspace = np.reshape(undersampled_kspace,
                                     (number_of_slices, number_of_coils, number_of_samples * new_ninterleaves),
                                     order='F')

    undersampled_spiral_kspace_dict = {'kspace': undersampled_kspace,
                                       'traj': undersampled_traj,
                                       'metadata': metadata}
    
    return undersampled_spiral_kspace_dict


def data_prep(fastMRI_path='/server/home/ncan/fastMRI', dtype='train', N=0, R=8, verbose=0):
    """
    Prepares data for train.py
    
    Args:
    fastMRI_path (str): Path to the fastMRI data directory.
    dtype (str): One of the three options: 'train', 'val', 'test'.
    N (int): Number of files to inclue in the data prep. If N=0, all files will be prepped.
    R (int): Acceleration factor.
    """
    data_dir = os.path.join(fastMRI_path, 'multicoil_' + dtype)
    undersampled_dir = os.path.join(fastMRI_path, 'undersampled_' + dtype)
    fully_sampled_dir = os.path.join(fastMRI_path, 'fully_sampled_' + dtype)
    
    # Check if directory exists, if not, create it
    if not os.path.exists(undersampled_dir):
        os.makedirs(undersampled_dir)
        print(f"Directory '{undersampled_dir}' created.")
    if not os.path.exists(fully_sampled_dir):
        os.makedirs(fully_sampled_dir)
        print(f"Directory '{fully_sampled_dir}' created.")
        
    data_names = np.array(os.listdir(data_dir))

    # Ensure that N is less than len(data_names) and that N=0 gets overwritten
    # as N = len(data_names)
    if N > len(data_names):
        warn('Note! Your value of N exceeds the number of files!' + 
                     '\nN has been set to: ' + str(len(data_names)))
        N = len(data_names)
        
    elif N == 0:
        print('PREPARING DATA FOR ALL ' + str(len(data_names)) + ' FILES.')
        N = len(data_names)

    else:
        print('Preparing data for ' + str(N) + ' file(s)...')

    # -------------------------------------------------------------------------
    # Data prep the first N files, skipping ones that have already been prepped
    # -------------------------------------------------------------------------
    for i in range(N):
        filename = data_names[i] # input file
        fully_sampled_filename = 'fully_sampled_' + filename[:-3] # output file
        undersampled_filename = 'undersampled_' + filename[:-3] # output file
        
        data_path = os.path.join(data_dir, filename) # input file
        fully_sampled_path = os.path.join(fully_sampled_dir, fully_sampled_filename)
        undersampled_path = os.path.join(undersampled_dir, undersampled_filename) 
        
        if not (os.path.exists(undersampled_path + '_kspace.h5') and \
                os.path.exists(fully_sampled_path + '_kspace.h5')):
            
            fastMRI_data = read_fastmri_data(data_path, verbose=verbose)
            spiral_kspace_dict = fastMRI_to_spiral(fastMRI_data, verbose=verbose)
            undersampled_spiral_kspace_dict = spiral_to_undersampled(spiral_kspace_dict, R)

            print('Writing file... ' + str(i+1) + '/' + str(N) + ' in progress.')
            write_kspace_data(spiral_kspace_dict, undersampled_spiral_kspace_dict,
                      fully_sampled_path, undersampled_path)


def read_kspace_data(fully_sampled_path, undersampled_path):
    """
    Reads spiral k-space data, trajectory, and header.
    
    Args:
    fully_sampled_path (str): FS path for spiral data.
    undersampled_path (str): US path for spiral data.

    Returns:
    full_kspace_data (dict): Dictionary holding fully sampled spiral k-space
        data, trajectory, and ISMRMRD header. keys are 'kspace', 'traj', and
        'metadata'

    undersampled_kspace_data (dict): Dictionary holding undersampled spiral
        k-space data, trajectory, and ISMRMRD header. keys are 'kspace',
        'traj', and 'metadata'
    """    
    # ------------------------------------------------------------------------
    # Read fully sampled kspace
    # ------------------------------------------------------------------------
    with h5py.File((fully_sampled_path + '_kspace.h5'), 'r') as f: # kspace
        kspace = f['kspace'][:]
    with h5py.File((fully_sampled_path + '_traj.h5'), 'r') as f: # traj
        traj = f['traj'][:]
    with open((fully_sampled_path + '_metadata.json'), 'r') as file: # metadata (ISMRMRD header)
        metadata = json.load(file)
    
    full_kspace_data = {'kspace': kspace, 'traj': traj, 'metadata': metadata}
    
    # ------------------------------------------------------------------------
    # Read undersampled kspace
    # ------------------------------------------------------------------------
    with h5py.File((undersampled_path + '_kspace.h5'), 'r') as f: # kspace
        kspace = f['kspace'][:]
    with h5py.File((undersampled_path + '_traj.h5'), 'r') as f: # traj
        traj = f['traj'][:]
    with open((undersampled_path + '_metadata.json'), 'r') as file: # metadata (ISMRMRD header)
        metadata = json.load(file)
    
    undersampled_kspace_data = {'kspace': kspace, 'traj': traj, 'metadata': metadata}

    return full_kspace_data, undersampled_kspace_data
    

def write_kspace_data(spiral_kspace_dict, undersampled_spiral_kspace_dict,
                      fully_sampled_path, undersampled_path):
    """
    to be written
    """
    # ------------------------------------------------------------------------
    # Write fully sampled kspace
    # ------------------------------------------------------------------------
    start_time = time.time()

    with h5py.File((fully_sampled_path + '_kspace.h5'), 'w') as f: # kspace
        f.create_dataset('kspace', data=spiral_kspace_dict['kspace'])
    with h5py.File((fully_sampled_path + '_traj.h5'), 'w') as f: # trajectory
        f.create_dataset('traj', data=spiral_kspace_dict['traj'])
    with open((fully_sampled_path + '_metadata.json'), 'w') as file: # metadata (ISMRMRD header)
        json.dump(spiral_kspace_dict['metadata'], file)
    
    print('Fully sampled data written!')
    print(f"{(time.time() - start_time):.1f} seconds have passed.")

    # ------------------------------------------------------------------------
    # Write undersampled kspace
    # ------------------------------------------------------------------------
    print('Writing the undersampled data...')
    start_time = time.time()
    
    with h5py.File((undersampled_path + '_kspace.h5'), 'w') as f: # kspace
        f.create_dataset('kspace', data=undersampled_spiral_kspace_dict['kspace'])
    with h5py.File((undersampled_path + '_traj.h5'), 'w') as f: # trajectory
        f.create_dataset('traj', data=undersampled_spiral_kspace_dict['traj'])
    with open((undersampled_path + '_metadata.json'), 'w') as file: # metadata (ISMRMRD header)
        json.dump(undersampled_spiral_kspace_dict['metadata'], file)

    print('Undersampled data written!')
    print(f"{(time.time() - start_time):.1f} seconds have passed.")

def fastMRI_to_spiral(fastMRI_data, verbose=1):
    """
    Augments a spiral acquisition by inverse-gridding the Cartesian k-space
    data provided by fastMRI into spiral k-space data. Utilizes the KbNufft
    library.
    
    Args:
    fastMRI_data (dict): Dictionary extracted from a fastMRI .h5 file.
    
    Returns:
    spiral_kspace_dict (dict): Dictionary holding spiral k-space data,
         k-space trajectory, and ISMRMRD header. keys are 'kspace', 'traj',
         and 'metadata'
    """
    from torch import cuda, device, from_numpy
    import torchkbnufft as tkbn
    from xmltodict import parse

    if cuda.is_available():
        device = device("cuda")
    else:
        device = device("cpu")
        
    # Load ISMRMRD header
    metadata = parse(fastMRI_data['metadata'].decode('utf-8'))

    # Load k-space & check if it is oversampled along readout
    if fastMRI_data['kspace'].shape[2] != fastMRI_data['kspace'].shape[3]:
        kspace = remove_readout_oversampling(fastMRI_data['kspace'], metadata)
    else:
        kspace = fastMRI_data['kspace']
    
    # Define matrix size for recon
    matrix_size = {'x': kspace.shape[-2], 'y': kspace.shape[-1]}
    im_size = (matrix_size['x'], matrix_size['y'])
    grid_size = (2*matrix_size['x'], 2*matrix_size['y'])
    
    # Create a kbInterp object, which is basically inverse gridding
    kb_ob = tkbn.KbInterp(im_size=im_size, grid_size=grid_size)
    
    # Generate spiral kspace trajectory using sigpy
    traj, metadata = generate_spiral_trajectory(metadata, verbose, ninterleaves=48)
    
    # Create batch dimension for traj
    batch_size = kspace.shape[0]
    traj_batch = np.stack([traj] * batch_size, axis=0)
    
    # ------------------------------------------------------------------------
    # Augment spiral kspace data
    # ------------------------------------------------------------------------
    
    # fft the cartesian kspace to a cartesian image
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

    # nufft the cartesian iamge to non-cartesian kspace
    kb_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
    spiral_kspace = kb_ob(from_numpy(image), from_numpy(traj_batch))

    # Convert spiral_kspace from torch.Tensor to numpy array
    kspace = spiral_kspace.numpy()
    
    # Put everything in a dict
    spiral_kspace_dict = {'kspace': kspace,
                          'traj': traj,
                          'metadata': metadata}
    
    return spiral_kspace_dict


def get_train_directory(args):
    """
    Parameters
    ----------
    args :  args.data_opt--dataset to be used in training & testing
    Note: users should set the directories prior to running train file
    Returns
    -------
    directories of the kspace, sensitivity maps and mask
    kspace and sensitivity maps should have size of nSlices x nrow x ncol x
    ncoil and mask should have size of nrow x ncol

    """

    if args.data_opt == 'Coronal_PD':

        kspace_dir = '...'
        coil_dir = '...'

    elif args.data_opt == 'Coronal_PDFS':

        kspace_dir = '...'
        coil_dir = '...'

    else:
        raise ValueError('Invalid data option')

    mask_dir = '...'

    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', 
          coil_dir, '\n \n mask dir: ', mask_dir)

    return kspace_dir, coil_dir, mask_dir


def generate_spiral_trajectory(metadata, verbose=1, ninterleaves=48):
    """
    Generates spiral trajectory using sigpy.mri.spiral
    
    Args:
    metadata (dict): ISMRMRD header extracted from fastMRI .h5 file
    verbose (bool): if True, plot the trajectory using visualize.traj
    ninterleaves (int): number of interleaves
    
    Returns:
    ndarray: spiral kspace trajectory
    """
    import sigpy.mri as mr
    
    # Get parameters
    fov_mm = float(metadata['ismrmrdHeader']['encoding']['reconSpace'] \
                   ['fieldOfView_mm']['x']) # fov in mm
    fov = fov_mm / 1000 # fov in meters
    N = float(metadata['ismrmrdHeader']['encoding']['reconSpace'] \
              ['matrixSize']['x']) # matrix size
    f_sampling = 1  # undersampling factor in freq encoding direction
    R = 1 # undersampling factor orthogonal to freq encoding direction
    alpha = 1 # Variable density factor
    gm = 40e-3 # Maximum gradient amplitude (T/m)
    sm = 150 # Maximum slew rate (T/m/s)
    gamma = 267800000 # Gyromagnetic ratio (rad/T/s)
    
    # Generate trajectory with units of cycles/FOV
    traj = mr.spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm, gamma)

    # Convert units to radians/voxel
    traj = traj * (2 * np.pi) * fov / N
        
    if verbose == 1:
        visualize.traj(traj, ninterleaves, gridsize=50, vmax=120, logscale=1, mincnt=18)

    traj = np.transpose(traj) # 2 x (Nk * Ni)

    metadata['spiralParameters'] = {'ninterleaves' : ninterleaves}
    
    return traj, metadata


def remove_readout_oversampling(kspace, metadata):
    """
    First removes 50% readout oversampling, then crops phase encode if needed to make square.
    Sanity checks matrix size with ISMRMRD header file to ensure that the matrix sizes match.
    
    Args:
        kspace: Complex array of shape (Ns, Nc, Nx, Ny) where Nx has 2x oversampling
        metadata: ISMRMRD header
        
    Returns:
        kspace_square: Complex array with square spatial dimensions after oversampling removal
    """
    Ns, Nc, Nx, Ny = kspace.shape
    
    # FFT to image space
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    
    # First remove readout oversampling
    readout_size = Nx // 2  # Remove 50% oversampling
    start_x = (Nx - readout_size) // 2
    image_no_os = image[:, :, start_x:start_x + readout_size, :]
    
    # Check if phase encode needs cropping to make square
    _, _, Nx_new, Ny = image_no_os.shape
    if Ny > Nx_new:  # If phase encode is longer after oversampling removal
        start_y = (Ny - Nx_new) // 2
        image_square = image_no_os[:, :, :, start_y:start_y + Nx_new]
    else:
        image_square = image_no_os
    
    # Back to k-space
    image_square = np.fft.ifftshift(image_square, axes=(-2, -1))
    kspace_square = np.fft.fft2(image_square, axes=(-2, -1))

    # Throws error if the matrix size doesn't match that of reconSpace in ISMRMRD header
    sanity.matrix_size_check(kspace_square, metadata)
    
    return kspace_square

    
def read_fastmri_data(file_path, verbose=1):
    """
    Reads FastMRI data from an H5 file.
    
    Args:
    file_path (str): Path to the H5 file.
    
    Returns:
    dict: A dictionary containing the available data from the H5 file.
    """
    data = {}
    
    with h5py.File(file_path, 'r') as hf:
        if verbose:
            # Print the keys to see what data is available in the file
            print("Keys in the H5 file:", list(hf.keys()))
        
        # Read the k-space data
        data['kspace'] = hf['kspace'][()]
        
        # Read the reconstruction (if available)
        if 'reconstruction_rss' in hf:
            data['reconstruction'] = hf['reconstruction_rss'][()]
        
        # Read the DICOM metadata (if available)
        if 'ismrmrd_header' in hf:
            data['metadata'] = hf['ismrmrd_header'][()]

    if verbose:
        # Print some information about the data
        print("K-space data shape:", data['kspace'].shape)
        if 'reconstruction' in data:
            print("Reconstruction shape:", data['reconstruction'].shape)
    
    return data


def imtile(images):
    """
    Tiles a list of grayscale images along the dimension with the fewest images.

    Args:
    images (list of np.ndarray): List of 2D numpy arrays representing grayscale images.

    Returns:
    np.ndarray: The tiled image.
    """
    if len(images) == 0:
        raise ValueError("No images to tile.")
    
    # Determine the minimum dimension (rows or columns)
    img_shape = np.array(images[0].shape)
    
    # Ensure all images are of the same size
    for img in images:
        if img.shape != images[0].shape:
            raise ValueError("All images must have the same dimensions.")
    
    # Calculate the number of rows and columns for the tiling
    num_images = len(images)
    
    # Tile images along the dimension with the fewest images (min of rows or cols)
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))
    
    # Create an empty array for the tiled image
    tiled_img = np.zeros((num_rows * img_shape[0], num_cols * img_shape[1]),
                         dtype=images[0].dtype)
    
    # Place images in the tiled array
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        tiled_img[row*img_shape[0]:(row+1)*img_shape[0],
                  col*img_shape[1]:(col+1)*img_shape[1]] = img
    
    return tiled_img