import sanity
import visualize

import numpy as np

import os
from warnings import warn
import time
import json
import h5py

import torchkbnufft as tkbn
from torch import cuda, device, from_numpy

def nufft_adj(kspace_data, which_fov="half"):
    """
    Kaiser-Bessel Adjoint NUFFT function.
    
    Args:
    kspace_data (dict): Dictionary holding non-Cartesian k-space data,
        trajectory, and ISMRMRD header. keys are 'kspace', 'traj', and
        'metadata'
    which_fov (str): Decides between reconstructing 'full' or 'half' fov. Note
        that data is oversampled in the readout direction.

    Returns:
    image (ndarray): blah blah blah
    """

    if which_fov == "full":
        reconSpace_size = kspace_data['metadata']['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['x']
        matrix_size = {'x': reconSpace_size * 2, 'y': reconSpace_size * 2} # full FOV is double that of the reconSpace FOV
    elif which_fov == "half":
        matrix_size = kspace_data['metadata']['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']
    else:
        raise ValueError('which_fov must be set to either "full" or "half"')

    # Define matrix size for recon
    im_size = (int(matrix_size['x']), int(matrix_size['y']))
    grid_size = (2*int(matrix_size['x']), 2*int(matrix_size['y']))

    # Get kspace and traj
    kspace = kspace_data['kspace']
    traj = kspace_data['traj'].astype(np.float32) # the astype is probably not necessary
    
    # Create batch dimension for traj
    batch_size = kspace.shape[0]
    traj_batch = np.stack([traj] * batch_size, axis=0)
    
    # Convert data from numpy.ndarray to torch.tensor
    kspace = from_numpy(kspace)
    traj_batch = from_numpy(traj_batch)

    # ------------------------------------------------------------------------
    # Calculate the density compensation function
    # ------------------------------------------------------------------------
    dcomp = tkbn.calc_density_compensation_function(from_numpy(traj),
                                                    im_size=im_size,
                                                    grid_size=grid_size)

    # Get tkbn adjoint nufft operator
    adjkb_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size);
    
    # Perform adjoint nufft
    image = adjkb_ob(kspace * dcomp, traj_batch);

    return image.numpy()


def spiral_to_undersampled(spiral_kspace_dict, R):
    """
    Undersamples the spiral kspace data with acceleration factor R. Undersampling
    is performed via zero-filling.
    
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
    Ns = kspace.shape[0] # number of slices
    Nc = sanity.num_coils_check(kspace, metadata) # number of coils
    Ni = metadata['spiralParameters']['ninterleaves'] # number of interleaves
    Nk = int(traj.shape[1]/Ni) # number of samples
    
    # Check if R is valid
    if Ni % R != 0:
        raise ValueError(f"Acceleration factor {R} must be a factor of number of interleaves {Ni}")
    if R > Ni:
        raise ValueError(f"Acceleration factor {R} cannot be larger than number of interleaves {Ni}")

    # -------------------------------------------------------------------------
    # Reshape kspace into Ns x Nc x Nk x Ni
    # -------------------------------------------------------------------------
    kspace = np.reshape(kspace, (Ns, Nc, Nk, Ni), order='F')

    # -------------------------------------------------------------------------
    # Create mask of which interleaves to keep
    # -------------------------------------------------------------------------
    mask = np.zeros(Ni, dtype=bool)
    mask[::R] = True

    # -------------------------------------------------------------------------
    # Zero-fill the undersampled regions
    # -------------------------------------------------------------------------
    undersampled_kspace = np.zeros_like(kspace)
    undersampled_kspace[:,:,:,mask] = kspace[:,:,:,mask]

    # -------------------------------------------------------------------------
    # Reshape them back into their original form
    # -------------------------------------------------------------------------
    # traj into 2 x (Nk * Ni)
    # kspace into Ns x Nc x (Nk * Ni)
    undersampled_kspace = np.reshape(undersampled_kspace, (Ns, Nc, Nk * Ni), order='F')

    undersampled_spiral_kspace_dict = {'kspace': undersampled_kspace,
                                       'traj': traj,
                                       'mask': mask,
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
    undersampled_dir = os.path.join(fastMRI_path, 'undersampled_' + dtype + '_R' + str(R))
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
    
    print('Fully sampled data written!' + f"{(time.time() - start_time):.1f} seconds have passed.")

    # ------------------------------------------------------------------------
    # Write undersampled kspace
    # ------------------------------------------------------------------------
    start_time = time.time()
    
    with h5py.File((undersampled_path + '_kspace.h5'), 'w') as f: # kspace
        f.create_dataset('kspace', data=undersampled_spiral_kspace_dict['kspace'])
    with h5py.File((undersampled_path + '_traj.h5'), 'w') as f: # trajectory
        f.create_dataset('traj', data=undersampled_spiral_kspace_dict['traj'])
    with open((undersampled_path + '_metadata.json'), 'w') as file: # metadata (ISMRMRD header)
        json.dump(undersampled_spiral_kspace_dict['metadata'], file)

    print('Undersampled data written!'+ f"{(time.time() - start_time):.1f} seconds have passed.")

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
    from xmltodict import parse
    from torch import cuda, device, from_numpy

    if cuda.is_available():
        device = device("cuda")
    else:
        device = device("cpu")
        
    # Load ISMRMRD header
    metadata = parse(fastMRI_data['metadata'].decode('utf-8'))

    # Load k-space
    kspace = fastMRI_data['kspace']

    # Go to image space
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

    # Zero pad image to square for simplicity
    if image.shape[2] != image.shape[3]:
        max_size = np.max([image.shape[2], image.shape[3]])
        image = pad_array(image, target_sizes=(max_size, max_size), pad_dims=(2, 3))
        
    # Generate spiral kspace trajectory using sigpy
    traj, metadata = generate_spiral_trajectory(metadata, verbose, ninterleaves=48)
    
    # Create batch dimension for traj
    batch_size = kspace.shape[0]
    traj_batch = np.stack([traj] * batch_size, axis=0)
    
    # ------------------------------------------------------------------------
    # Augment spiral kspace data
    # ------------------------------------------------------------------------
    # Define matrix size for recon
    matrix_size = {'x': image.shape[-2], 'y': image.shape[-1]}
    im_size = (matrix_size['x'], matrix_size['y'])
    grid_size = (2 * matrix_size['x'], 2 * matrix_size['y'])

    # nufft the cartesian image to non-cartesian kspace
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
    fov_mm = float(metadata['ismrmrdHeader']['encoding']['encodedSpace'] \
                   ['fieldOfView_mm']['x']) # fov in mm
    fov = fov_mm / 1000 # fov in meters
    N = float(metadata['ismrmrdHeader']['encoding']['encodedSpace'] \
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

def pad_array(array, target_sizes, pad_dims=None):
    """
    Zero pad an N-dimensional array to target sizes along specified dimensions.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array of any dimensionality
    target_sizes : tuple or list
        Target sizes for each dimension to be padded. Must have same length
        as pad_dims if pad_dims is specified, otherwise must be length equal
        to array.ndim
    pad_dims : tuple or list, optional
        Indices of dimensions to pad (0 to array.ndim-1). If None, all 
        dimensions are padded.
        
    Returns:
    --------
    numpy.ndarray
        Padded array
        
    Examples:
    --------
    # Pad a 2D array
    img_2d = np.random.rand(28, 28)
    padded_2d = pad_array(img_2d, target_sizes=(32, 32))
    
    # Pad specific dimensions of a 3D array
    volume = np.random.rand(16, 32, 24)
    padded_3d = pad_array(volume, target_sizes=(48, 32), pad_dims=(0, 2))
    
    # Pad a 4D array
    tensor_4d = np.random.rand(16, 32, 48, 64)
    padded_4d = pad_array(tensor_4d, target_sizes=(32, 64, 96, 128))
    
    # Pad a 5D array (e.g., batch of 4D images)
    batch = np.random.rand(8, 16, 32, 48, 64)
    padded_5d = pad_array(batch, target_sizes=(16, 32, 64, 96, 128))
    """
    
    # Validate input is numpy array
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # If pad_dims not specified, pad all dimensions
    if pad_dims is None:
        pad_dims = tuple(range(array.ndim))
        if len(target_sizes) != array.ndim:
            raise ValueError(
                f"Must specify target sizes for all {array.ndim} dimensions "
                "when pad_dims is None"
            )
    
    # Validate pad_dims and target_sizes
    if len(pad_dims) != len(target_sizes):
        raise ValueError("Number of target sizes must match number of dimensions to pad")
    
    if not all(0 <= dim < array.ndim for dim in pad_dims):
        raise ValueError(f"Pad dimensions must be between 0 and {array.ndim-1}")
    
    if not all(size >= array.shape[dim] for size, dim in zip(target_sizes, pad_dims)):
        smaller_dims = [
            f"dim{dim}({array.shape[dim]}->{size})" 
            for size, dim in zip(target_sizes, pad_dims) 
            if size < array.shape[dim]
        ]
        raise ValueError(
            "Target sizes must be greater than or equal to current sizes. "
            f"Problem with dimensions: {', '.join(smaller_dims)}"
        )
    
    # Calculate padding for each dimension
    pad_width = [(0, 0)] * array.ndim  # Initialize with no padding
    
    for dim, target in zip(pad_dims, target_sizes):
        current_size = array.shape[dim]
        total_pad = target - current_size
        # Calculate padding for before and after
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width[dim] = (pad_before, pad_after)
    
    # Apply padding
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    
    return padded_array

def normalize_array(arr):
    """
    Normalize any n-dimensional numpy array to range [0, 1].
    
    Parameters:
        arr (numpy.ndarray): Input array of any shape and dimension
        
    Returns:
        numpy.ndarray: Normalized array of same shape as input
        
    Example:
        >>> x = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalize_array(x)
        array([[0. , 0.2, 0.4],
               [0.6, 0.8, 1. ]])
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Handle edge case where max and min are the same
    if arr_max == arr_min:
        return np.zeros_like(arr)
        
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized