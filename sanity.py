def num_slices_check(kspace, metadata):
    """
    Compares k-space number of slices to that of the metadata (a.k.a. header)
    
    Parameters:
    kspace (numpy.ndarray): k-space data
    metadata (dict): ISMRMRD header converted to Python dictionary

    Returns:
    (int): number of slices
    """
    metadata_Ns = int(metadata['ismrmrdHeader']['encoding']['encodingLimits']['slice']['maximum']) + 1
    kspace_Ns = kspace.shape[0]

    if not metadata_Ns == kspace_Ns:
        error_msg = 'The k-space number of slices and metadata (header) number ' + \
                ' of coils do not match! \nkspace: ' + str(kspace_Ns) + \
                '\nMetadata: ' + str(metadata_Ns)
            
        raise Exception(error_msg)
    return int(kspace_Ns)

def num_coils_check(kspace, metadata):
    """
    Compares k-space number of coils to that of the metadata (a.k.a. header)
    
    Parameters:
    kspace (numpy.ndarray): k-space data
    metadata (dict): ISMRMRD header converted to Python dictionary

    Returns:
    (int): number of coils
    """
    metadata_Nc = int(metadata['ismrmrdHeader']['acquisitionSystemInformation']['receiverChannels'])
    kspace_Nc = kspace.shape[1]

    if not metadata_Nc == kspace_Nc:
        error_msg = 'The k-space number of coils and metadata (header) number ' + \
                ' of coils do not match! \nkspace: ' + str(kspace_Nc) + \
                '\nMetadata: ' + str(metadata_Nc)
            
        raise Exception(error_msg)
    return int(kspace_Nc)

def matrix_size_check(kspace, metadata):
    """
    Compares k-space matrix size to that of the metadata (a.k.a. header)
    
    Parameters:
    kspace (numpy.ndarray): k-space data
    metadata (dict): ISMRMRD header converted to Python dictionary

    Returns:
    (dict): matrix size
    """
    
    metadata_matrix_size = (int(metadata['ismrmrdHeader']['encoding'] \
                                ['reconSpace']['matrixSize']['x']),
                            int(metadata['ismrmrdHeader']['encoding'] \
                                ['reconSpace']['matrixSize']['y']))

    kspace_matrix_size = (kspace.shape[2], kspace.shape[3])

    if kspace_matrix_size != metadata_matrix_size:
        error_msg = 'The k-space matrix size and metadata (header) matrix ' + \
            ' size do not match! \nkspace: ' + str(kspace_matrix_size) + \
            '\nMetadata (reconSpace): ' + str(metadata_matrix_size)
        raise Exception(error_msg)
    
    return {'x' : kspace_matrix_size[0], 'y' : kspace_matrix_size[1]}