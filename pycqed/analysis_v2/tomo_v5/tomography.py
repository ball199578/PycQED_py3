import logging
import numpy as np
import qutip as qtp
from shots_packing import reshape_block, all_repetitions, get_segments_average

prerotations_dict = {'i': qtp.identity(2),
                    'x180': qtp.sigmax(),
                    'y90': qtp.rotation(qtp.sigmay(), np.pi / 2),
                    'ym90': qtp.rotation(qtp.sigmay(), -np.pi / 2),
                    'x90': qtp.rotation(qtp.sigmax(), np.pi / 2),
                    'xm90': qtp.rotation(qtp.sigmax(), -np.pi / 2)}
pauli_operators_dict = {'i': qtp.identity(2),
                        'z': qtp.sigmaz(),
                        'x': qtp.sigmax(),
                        'y': qtp.sigmay()}

def ordering_shots(shots_data, segments_per_block=64, block_size=4094):
    """
    Input:
        shots_data: (SxN_shots float) data
    segments_per_block: (int) Segments in sequence (pre-rotations + cal_points)
    block_size: (int) Size of block of shots acquired (detector.nr_shots for UHFQC)

    S = individual channels + correlation channels
    N_shots = number of total shots

    Output data:
    SxMxN_runs table
    M = pre-rotations + residual
    N_runs = number of runs of entire M segments
    """
    num_channels, N_shots = shots_data.shape[0]
    num_blocks = N_shots//block_size
    shots_packed = np.zeros((num_channels, num_blocks, segments_per_block))

    for s in range(num_channels):
        shots_packed[s,:,:] = get_segments_average(shots_data[,:],
                                                   segments_per_block=segments_per_block,
                                                   block_size=block_size,
                                                   average=False)

    return shots_packed

def default_threshold_algorithm(shots_init, mmt_thresholds):
    """
    Input:
    Output:
    """
    # threshold vector needs to be same dimension as channels
    assert(len(mmt_thresholds)==shots.shape[0])
    return np.where(shots_init<mmt_thresholds, True, False)

def tomography_multiplexed(data,
                           shots_mode=False,     # SHOTS PACKING KW
                           segments_per_block=64,
                           block_size=3968,
                           mmt_initialize=False, # MMT INIT KW
                           mmt_threshold_call='default',
                           mmt_thresholds=None,
                           num_qubits=2, # EXPERIMENT KW
                           re_vec=None, # BETA-RO KW
                           calpoints_repetitions=7,
                           prerotations_per_qubit=6, # PRE-ROT KW
                           prerotations_order=['i','x180',
                                               'y90','ym90',
                                               'x90','xm90'],
                           order_pauli_operator = ['i','z', # REPRESENTATION KW
                                                   'x', 'y']):
    # 1.0 Preparing data
    if shots_mode:
        # 1.1 Ordering shots
        if mmt_initialize:
            # if there is initialization, segments_per_block double
            # (for init and final mmt)
            segments_per_block = 2*segments_per_block

        shots_data = ordering_shots(shots_data=data,
                                    segments_per_block=segments_per_block,
                                    block_size=block_size)
        # 1.1.1 Thresholding data shots

        # 1.1.2 Computing correlation channels

        # 1.1.3 Post-selecting on init shots
        if mmt_initialize:
            # default thresholder call and threshold values
            if mmt_threshold_call == 'default':
                mmt_threshold_call = default_threshold_algorithm
            if mmt_thresholds is None:
                mmt_thresholds = np.zeros(shots_data.shape[0])

            mask_data = mmt_threshold_call(shots_data[::2],mmt_thresholds)
        else:
            mask_data = np.ones(shots_data.shape,dtype=np.bool)
        shots_filtered = shots_data[mask_data]

        # 1.1.4 Thresholding
        raise NotImplementedError('Still under construction. Thresholding to be implemented')
        # 1.1.5 Averaging
        data_avg = np.nanmean(shots_filtered, axis=2)
    else:
        data_avg = data

    #data_avg is SxM (S=channels, M=segments)
    total_num_prerotations = prerotations_per_qubit**num_qubits # 6**2
    data_cals = data_avg[:,total_num_prerotations:]
    # 1.2 Offset substraction
    offsets = np.mean(data_cals, axis=1)
    data_avg -= offsets
    data_cals = data_avg[:,total_num_prerotations:]

    # HERE COMES THE PART WHERE QUBIT ORDERING GETS IMPORTANT.

    # 1.3 Signal scaling
    # 1.3.1 Signal scaling by dominant betas
    num_channels = dava_avg.shape[0]
    mmt_operators = get_mmt_operators(num_qubits=num_qubits)

    beta_matrix = construct_beta_matrix(num_qubits, mmt_operators, re_vec)
    beta_matrix_inv = np.linalg.inv(beta_matrix)
    dom_beta = np.zeros(num_channels)
    for s in range(num_channels):
        cal_points = get_avg_cal_points(data_cals[s,:],
                                        repetitions=calpoints_repetitions)
        betas = np.dot(beta_matrix_inv, cal_points)
        dom_beta_idx = np.argmax(abs(betas))
        dom_beta[s] = betas[dom_beta_idx]
        data_avg[s,:] = data_avg[s,:] / dom_beta[s]

    data_cals = data_avg[:,total_num_prerotations:]

    # 1.3.2 Noise scaling
    # get avg-std for each channel
    avg_std = np.zeros(num_channels)
    for s in range(num_channels):
        cal_points_std = get_std_cal_points(data_cals[s,:],
                                            repetitions=calpoints_repetitions)
        avg_std[s] = np.mean(cal_points_std)
    # scale(divide) each channel by avg-std/avg-avg-std
    std_scale = np.mean(avg_std)
    for s in range(num_channels):
        data_avg[s,:] *= std_scale/avg_std[s]

    data_prerot = data_avg[:,:total_num_prerotations]
    prerot_data_stacked = data_prerot.flatten() #put a 'F' for other order
    data_cals = data_avg[:,total_num_prerotations:]

    # 2.0 Performing the inversion
    # 2.1 Calculate adjusted betas
    beta_matrix = construct_beta_matrix(num_qubits, mmt_operators, re_vec)
    beta_matrix_inv = np.linalg.inv(beta_matrix)
    betas_vec = np.zeros((num_channels, beta_matrix.shape[1]))
    for s in range(data_avg.shape[0]):
        cal_points = get_avg_cal_points(data_cals[s,:],
                                        repetitions=calpoints_repetitions)
        betas_vec[s,:] = np.dot(beta_matrix_inv, cal_points)

    # 2.2 Get the matrix for pre-rotations
    # prerotations_operators sets order for data segments
    prerotations_operators = get_prerotation_operators(num_qubits, prerotations_order)
    # order_pauli_operators sets order for matrix (and therefore pauli vector).

    # combine all basis in the given order
    pauli_basis = get_basis_operators(num_qubits, order_pauli_operators)

    # Assemble matrix.
    coefficient_matrix = np.zeros((num_channels * total_num_prerotations ** num_qubits, 4 ** num_qubits))
    # err_coefficient_matrix = np.zeros((num_channels * total_num_prerotations ** num_qubits, 4 ** num_qubits))

    for ch in range(num_channels):
        for rotation_index,rotation_operator in enumerate(prerotations_operators):
            for beta_index,mmt_op in enumerate(mmt_operators):
                (place, sign) = get_basis_index_from_rotation(mmt_op,
                                                              rotation_operator,
                                                              pauli_basis)
                coefficient_matrix[
                    (total_num_prerotations ** num_qubits) * ch + rotation_index, place] = sign * betas_vec[ch,beta_index]
                # err_coefficient_matrix[
                #     (total_num_prerotations ** num_qubits) * ch + rotation_index, place] = err_betas[ch][beta_index]

    # 2.3 Invert it.
    invert_tomo_matrix = np.linalg.pinv(coefficient_matrix)
    pauli_ops = np.dot(invert_tomo_matrix,prerot_data_stacked)
    return pauli_ops

# functions needed
def get_avg_cal_points(cal_points_vec, repetitions=7):
    avg_points = [np.mean(cal_points_vec[calpoints_repetitions*i:calpoints_repetitions*(i+1)])
                  for i in range(2**num_qubits)]
    return np.array(avg_points)

def get_std_cal_points(cal_points_vec, repetitions=7):
    avg_points = [np.std(cal_points_vec[calpoints_repetitions*i:calpoints_repetitions*(i+1)])
                  for i in range(2**num_qubits)]
    return np.array(avg_points)

def construct_beta_matrix(num_qubits, mmt_operators, re_vec=None):
    """
    Input:
    Calibration state
    RE probability

    Output:
    elements for ALL MMT operators
    """
    if re_vec is None:
        re_vec = np.zeros(num_qubits)
    thermal_populations = get_re_populations(num_qubits=num_qubits,
                                             re_vec=re_vec)
    rho_ground = qtp.Qobj(np.diag(thermal_populations),
                          dims=[[2]*num_qubits,[2]*num_qubits])
    cal_prerotations = get_cal_prerotations(num_qubits=num_qubits)

    betas_matrix = np.zeros((len(cal_prerotations),len(mmt_operators)))
    for id_calstate, pre_rot in enumerate(cal_prerotations):
        cal_state = pre_rot.dag()*rho_ground*pre_rot
        for id_op, mmt_op in enumerate(mmt_operators):
            betas_matrix[id_calstate,id_op] = np.real((cal_state*mmt_op).tr())
    return betas_matrix

def get_mmt_operators(num_qubits):
    """
    Input:
    num_qubits (int)

    Output:
    list with qutip operators for each term in beta expansion.
    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    operators_list = [qtp.identity(2),qtp.sigmaz()]
    cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                     operators_list for op_cum in operators_list]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                             operators_list for op_cum in cumulative_op]
    return cumulative_op

def get_cal_prerotations(num_qubits):
    """
    Input:
    num_qubits (int)

    Output:
    list with qutip operators for rotating ground state into a cal point.
    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    operators_list = [qtp.identity(2),qtp.sigmax()]
    cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                     operators_list for op_cum in operators_list]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                             operators_list for op_cum in cumulative_op]
    return cumulative_op

def get_re_populations(num_qubits, re_vec):
    """
    Input:
    num_qubits (int)
    re_vec (float array)

    Output:
    vector with populations for thermal ground state.
    NOTE:
    The order in which we construct cumulative_pops ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    pop_list_0 = [1-re_vec[0], re_vec[0]]
    pop_list_1 = [1-re_vec[1], re_vec[1]]
    cumulative_pops = [this_pop1*this_pop2 for this_pop1 in
                       pop_list_0 for this_pop2 in pop_list_1]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            pop_list_i = [1-re_vec[i], re_vec[i]]
            cumulative_pops = [this_pop1*this_cum_pop for this_pop1 in
                               pop_list_i for this_cum_pop in cumulative_pops]
    # Redefining trace 1
    cumulative_pops[0] -= (np.sum(cumulative_pops)-1)
    return cumulative_pops

def get_data_prerotations(num_qubits, prerotations_order):
    """
    Input:
    num_qubits (int)

    Output:
    list with qutip operators for rotating ground state into a cal point.

    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                     prerotations_order for op_cum in prerotations_order]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                             prerotations_order for op_cum in cumulative_op]
    return cumulative_op

def get_basis_operators(num_qubits, order_pauli_operators):
    """
    Input:
    num_qubits (int)

    Output:
    list with qutip operators for rotating ground state into a cal point.

    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan.
    """
    pauli_operators_vec = [pauli_operators_dict[o] for o in order_pauli_operators]
    cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                     pauli_operators_vec for op_cum in pauli_operators_vec]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                             pauli_operators_vec for op_cum in cumulative_op]
    return cumulative_op

def get_prerotation_operators(num_qubits, order_pauli_operators):
    """
    Input:
    num_qubits (int)

    Output:
    list with qutip operators for rotating ground state into a cal point.

    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    pauli_operators_vec = [prerotations_dict[o] for o in order_pauli_operators]
    cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                     pauli_operators_vec for op_cum in pauli_operators_vec]
    if num_qubits>2:
        for qubit in range(num_qubits-2):
            cumulative_op = [qtp.tensor(this_op, op_cum) for this_op in
                             pauli_operators_vec for op_cum in cumulative_op]
    return cumulative_op

def get_basis_index_from_rotation(readout_operator, rotation_operator, measurement_basis):
    """
    Returns the position and sign of one of the betas in the coefficient
    matrix by checking to which basis matrix the readout matrix is mapped
    after rotation
    This is used in _calculate_coefficient_matrix
    """
    m = rotation_operator.dag() * readout_operator * rotation_operator
    for basis_index, basis in enumerate(measurement_basis):
        if(m == basis):
            return (basis_index, 1)
        elif(m == -basis):
            return (basis_index, -1)
    # if no basis is found raise an error
    raise Exception(
        'No basis vector found corresponding to the measurement rotation. Check that you have used Clifford Gates!')

def get_operator_from_string(op_string):
    """
    NOTE:
    The order in which we call qtp.tensor ensures LSQ is the right-most qubit
    This is in agreement with Device object convention used by Adriaan
    """
    for i,o in enumerate(op_string):
        if i==0:
            cum_op = pauli_operators[o]
        else:
            cum_op = qtp.tensor(cum_op,pauli_operators[o])
    return cum_op
