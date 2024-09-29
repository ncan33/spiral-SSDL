import numpy as np
#from skimage.measure import compare_ssim
import h5py


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

    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', coil_dir, '\n \n mask dir: ', mask_dir)

    return kspace_dir, coil_dir, mask_dir

def read_fastmri_data(file_path):
    """
    Reads FastMRI data from an H5 file.
    
    Args:
    file_path (str): Path to the H5 file.
    
    Returns:
    dict: A dictionary containing the available data from the H5 file.
    """
    data = {}
    
    with h5py.File(file_path, 'r') as hf:
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
    tiled_img = np.zeros((num_rows * img_shape[0], num_cols * img_shape[1]), dtype=images[0].dtype)
    
    # Place images in the tiled array
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        tiled_img[row*img_shape[0]:(row+1)*img_shape[0], col*img_shape[1]:(col+1)*img_shape[1]] = img
    
    return tiled_img