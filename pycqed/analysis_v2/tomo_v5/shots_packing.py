import numpy as np

def reshape_block(shots_data, segments_per_block=16, block_size=4092, mode='truncate'):
    """
    inputs: shots_data 1D array of dimension N
    organizes data in blocks of dimension block_size.
    num of blocks is N/block_size
    """
    N = len(shots_data)
    # Data dimension needs to be an integer multiple of block_size
    assert(N%block_size==0)
    num_blocks = N//block_size
    full_segments = block_size//segments_per_block
    orfan_segments = block_size % segments_per_block
    missing_segments = segments_per_block - orfan_segments
#     print(N,num_blocks,full_segments,orfan_segments,missing_segments)
    reshaped_data = shots_data.reshape((num_blocks,block_size))
    if mode.lower()=='truncate':
        truncate_idx = full_segments*segments_per_block
        return reshaped_data[:,:truncate_idx]
    elif mode.lower()=='padd':
        padd_dim = (full_segments+1)*segments_per_block
        return_block = np.nan*np.ones((num_blocks,padd_dim))
        return_block[:,:block_size] = reshaped_data
        return return_block
    else:
        raise ValueError('Mode not understood. Needs to be truncate or padd')

def all_repetitions(shots_data,segments_per_block=16):
    flat_dim = shots_data.shape[0]*shots_data.shape[1]
    # Data dimension needs to divide the segments_per_block
    assert(flat_dim%segments_per_block==0)
    num_blocks = flat_dim // segments_per_block
    block_data = shots_data.reshape((num_blocks,segments_per_block))
    return block_data

def get_segments_average(shots_data, segments_per_block=16, block_size=4092, mode='truncate', average=True):
    reshaped_data = reshape_block(shots_data=shots_data,
                                      segments_per_block=segments_per_block,
                                      block_size=block_size,
                                      mode=mode)
    all_reps = all_repetitions(shots_data=reshaped_data,
                                       segments_per_block=segments_per_block)
    if average:
        return np.mean(all_reps,axis=0)
    else:
        return all_reps