from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter




class NoiseParametersCZ(Instrument):
    '''
    Noise and other parameters for cz_superoperator_simulation_new
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)


        # Noise parameters
        self.add_parameter('T1_q0', unit='s',
                           label='T1 fluxing qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T1_q1', unit='s',
                           label='T1 static qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_q1', unit='s',
                           label='T2 static qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_q0_amplitude_dependent', unit='Hz, a.u., s',
                           label='fitcoefficients giving T2echo_q0 as a function of frequency_q0: gc, amp, tau. Function is gc+gc*amp*np.exp(-x/tau)',
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())   # , initial_value=np.array([-1,-1,-1])
        # for flux noise simulations
        self.add_parameter('sigma_q0', unit='flux quanta',
                           label='standard deviation of the Gaussian from which we sample the flux bias, q0',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('sigma_q1', unit='flux quanta',
                           label='standard deviation of the Gaussian from which we sample the flux bias, q1',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        # Some system parameters
        self.add_parameter('w_bus', unit='Hz',
                           label='omega of the bus resonator',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('alpha_q1', unit='Hz',
                           label='anharmonicity of the static qubit',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('w_q1_sweetspot',
                           label='NB: different from the operating point in general',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('Z_rotations_length', unit='s',
                           label='duration of the single qubit Z rotations at the end of the pulse',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        
        
        # Control parameters for the simulations
        self.add_parameter('dressed_compsub',
                           label='true if we use the definition of the comp subspace that uses the dressed 00,01,10,11 states',
                           parameter_class=ManualParameter,
                           vals=vals.Bool())
        self.add_parameter('distortions',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('voltage_scaling_factor', unit='a.u.',
                           label='scaling factor for the voltage for a CZ pulse',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('n_sampling_gaussian_vec',
                           label='array. each element is a number of samples from the gaussian distribution. Std to guarantee convergence is [11]. More are used only to verify convergence',
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())
        self.add_parameter('cluster',
                           label='true if we want to use the cluster',
                           parameter_class=ManualParameter,
                           vals=vals.Bool())
        self.add_parameter('look_for_minimum',
                           label='changes cost function to optimize either research of minimum of avgatefid_pc or to get the heat map in general',
                           parameter_class=ManualParameter,
                           vals=vals.Bool())

        self.add_parameter('T2_scaling', unit='a.u.',
                           label='scaling factor for T2_q0_amplitude_dependent',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())


        





















