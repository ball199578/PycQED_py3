import lmfit
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
import logging
from scipy.stats import sem
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.utilities.general import SafeFormatter, format_value_string
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from matplotlib import colors as c


class RandomizedBenchmarking_SingleQubit_Analysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idx: int =1,
                 ignore_f_cal_pts: bool=False, **kwargs
                 ):
        """
        Analysis for single qubit randomized benchmarking.

        For basic options see docstring of BaseDataAnalysis

        Args:
            classification_method ["rates", ]   sets method to determine
                populations of g,e and f states. Currently only supports "rates"
                    rates: uses calibration points and rate equation from
                        Asaad et al. to determine populations
            rates_ch_idx (int) : sets the channel from which to use the data
                for the rate equations
            ignore_f_cal_pts (bool) : if True, ignores the f-state calibration
                points and instead makes the approximation that the f-state
                looks the same as the e-state in readout. This is useful when
                the ef-pulse is not calibrated.
        """
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs,
                         do_fitting=True, **kwargs)
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idx = rates_ch_idx
        self.d1 = 2
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-6:2]
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_zero'] = OrderedDict()
            self.raw_data_dict['cal_pts_one'] = OrderedDict()
            self.raw_data_dict['cal_pts_two'] = OrderedDict()
            self.raw_data_dict['measured_values_I'] = OrderedDict()
            self.raw_data_dict['measured_values_X'] = OrderedDict()
            for i, val_name in enumerate(a.value_names):

                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*2)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')

                self.raw_data_dict['binned_vals'][val_name] = binned_yvals
                self.raw_data_dict['cal_pts_zero'][val_name] =\
                    binned_yvals[-6:-4, :].flatten()
                self.raw_data_dict['cal_pts_one'][val_name] =\
                    binned_yvals[-4:-2, :].flatten()

                if self.ignore_f_cal_pts:
                    self.raw_data_dict['cal_pts_two'][val_name] =\
                        self.raw_data_dict['cal_pts_one'][val_name]
                else:
                    self.raw_data_dict['cal_pts_two'][val_name] =\
                        binned_yvals[-2:, :].flatten()

                self.raw_data_dict['measured_values_I'][val_name] =\
                    binned_yvals[:-6:2, :]
                self.raw_data_dict['measured_values_X'][val_name] =\
                    binned_yvals[1:-6:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        for key in ['V0', 'V1', 'V2', 'SI', 'SX', 'P0', 'P1', 'P2', 'M_inv']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            V0 = np.nanmean(
                self.raw_data_dict['cal_pts_zero'][val_name])
            V1 = np.nanmean(
                self.raw_data_dict['cal_pts_one'][val_name])
            V2 = np.nanmean(
                self.raw_data_dict['cal_pts_two'][val_name])

            self.proc_data_dict['V0'][val_name] = V0
            self.proc_data_dict['V1'][val_name] = V1
            self.proc_data_dict['V2'][val_name] = V2

            SI = np.nanmean(
                self.raw_data_dict['measured_values_I'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_X'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            P0, P1, P2, M_inv = populations_using_rate_equations(
                SI, SX, V0, V1, V2)
            self.proc_data_dict['P0'][val_name] = P0
            self.proc_data_dict['P1'][val_name] = P1
            self.proc_data_dict['P2'][val_name] = P2
            self.proc_data_dict['M_inv'][val_name] = M_inv

        classifier = logisticreg_classifier_machinelearning(
            self.proc_data_dict['cal_pts_zero'],
            self.proc_data_dict['cal_pts_one'],
            self.proc_data_dict['cal_pts_two'])
        self.proc_data_dict['classifier'] = classifier

        if self.classification_method == 'rates':
            val_name = self.raw_data_dict['value_names'][self.rates_ch_idx]
            self.proc_data_dict['M0'] = self.proc_data_dict['P0'][val_name]
            self.proc_data_dict['X1'] = 1-self.proc_data_dict['P2'][val_name]
        else:
            raise NotImplementedError()

    def run_fitting(self):
        super().run_fitting()

        leak_mod = lmfit.Model(leak_decay, independent_vars='m')
        leak_mod.set_param_hint('A', value=.95, min=0, vary=True)
        leak_mod.set_param_hint('B', value=.1, min=0, vary=True)

        leak_mod.set_param_hint('lambda_1', value=.99, vary=True)
        leak_mod.set_param_hint('L1', expr='(1-A)*(1-lambda_1)')
        leak_mod.set_param_hint('L2', expr='A*(1-lambda_1)')
        leak_mod.set_param_hint(
            'L1_cz', expr='1-(1-(1-A)*(1-lambda_1))**(1/1.5)')
        leak_mod.set_param_hint(
            'L2_cz', expr='1-(1-(A*(1-lambda_1)))**(1/1.5)')

        params = leak_mod.make_params()
        try:
            fit_res_leak = leak_mod.fit(data=self.proc_data_dict['X1'],
                                        m=self.proc_data_dict['ncl'],
                                        params=params)
            self.fit_res['leakage_decay'] = fit_res_leak
            lambda_1 = fit_res_leak.best_values['lambda_1']
            L1 = fit_res_leak.params['L1'].value
        except Exception as e:
            logging.warning("Fitting failed")
            logging.warning(e)
            lambda_1 = 1
            L1 = 0
            self.fit_res['leakage_decay'] = {}

        fit_res_rb = self.fit_rb_decay(lambda_1=lambda_1, L1=L1, simple=False)
        self.fit_res['rb_decay'] = fit_res_rb
        fit_res_rb_simple = self.fit_rb_decay(lambda_1=1, L1=0, simple=True)
        self.fit_res['rb_decay_simple'] = fit_res_rb_simple

        fr_rb = self.fit_res['rb_decay'].params
        fr_rb_simple = self.fit_res['rb_decay_simple'].params
        fr_dec = self.fit_res['leakage_decay'].params

        text_msg = 'Summary: \n'
        text_msg += format_value_string(r'$\epsilon_{{\mathrm{{simple}}}}$',
                                        fr_rb_simple['eps'], '\n')
        text_msg += format_value_string(r'$\epsilon_{{X_1}}$',
                                        fr_rb['eps'], '\n')
        text_msg += format_value_string(r'$L_1$', fr_dec['L1'], '\n')
        text_msg += format_value_string(r'$L_2$', fr_dec['L2'], '\n')

        self.proc_data_dict['rb_msg'] = text_msg

    def fit_rb_decay(self, lambda_1: float, L1: float, simple: bool=False):
        """
        Fits the data
        """
        fit_mod_rb = lmfit.Model(full_rb_decay, independent_vars='m')
        fit_mod_rb.set_param_hint('A', value=.5, min=0, vary=True)
        if simple:
            fit_mod_rb.set_param_hint('B', value=0, vary=False)
        else:
            fit_mod_rb.set_param_hint('B', value=.1, min=0, vary=True)
        fit_mod_rb.set_param_hint('C', value=.4, min=0, max=1, vary=True)

        fit_mod_rb.set_param_hint('lambda_1', value=lambda_1, vary=False)
        fit_mod_rb.set_param_hint('lambda_2', value=.95, vary=True)

        # d1 = dimensionality of computational subspace
        fit_mod_rb.set_param_hint('d1', value=self.d1, vary=False)
        fit_mod_rb.set_param_hint('L1', value=L1, vary=False)

        # Note that all derived quantities are expressed directly in
        fit_mod_rb.set_param_hint(
            'F', expr='1/d1*((d1-1)*lambda_2+1-L1)', vary=True)
        fit_mod_rb.set_param_hint('eps',
                                  expr='1-(1/d1*((d1-1)*lambda_2+1-L1))')
        # Only valid for single qubit RB assumption equal error rates
        fit_mod_rb.set_param_hint(
            'F_g', expr='(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)')
        fit_mod_rb.set_param_hint(
            'eps_g', expr='1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.875)')
        # Only valid for two qubit RB assumption all error in CZ
        fit_mod_rb.set_param_hint(
            'F_cz', expr='(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)')
        fit_mod_rb.set_param_hint(
            'eps_cz', expr='1-(1/d1*((d1-1)*lambda_2+1-L1))**(1/1.5)')

        params = fit_mod_rb.make_params()
        fit_res_rb = fit_mod_rb.fit(data=self.proc_data_dict['M0'],
                                    m=self.proc_data_dict['ncl'],
                                    params=params)

        return fit_res_rb

    def prepare_plots(self):
        val_names = self.raw_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['bins'],
                'yvals': np.nanmean(self.raw_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.raw_data_dict['binned_vals'][val_name], axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.raw_data_dict['value_units'][i],
                'title': self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'],
            }

        fs = plt.rcParams['figure.figsize']
        self.plot_dicts['cal_points_hexbin'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.raw_data_dict['cal_pts_zero'][val_names[0]],
                        self.raw_data_dict['cal_pts_zero'][val_names[1]]),
            'shots_1': (self.raw_data_dict['cal_pts_one'][val_names[0]],
                        self.raw_data_dict['cal_pts_one'][val_names[1]]),
            'shots_2': (self.raw_data_dict['cal_pts_two'][val_names[0]],
                        self.raw_data_dict['cal_pts_two'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.raw_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.raw_data_dict['value_units'][1],
            'title': self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'] + ' hexbin plot',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        for i, val_name in enumerate(val_names):
            self.plot_dicts['raw_RB_curve_data_{}'.format(val_name)] = {
                'plotfn': plot_raw_RB_curve,
                'ncl': self.proc_data_dict['ncl'],
                'SI': self.proc_data_dict['SI'][val_name],
                'SX': self.proc_data_dict['SX'][val_name],
                'V0': self.proc_data_dict['V0'][val_name],
                'V1': self.proc_data_dict['V1'][val_name],
                'V2': self.proc_data_dict['V2'][val_name],

                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string']+'\n'+self.proc_data_dict['measurementstring'],
            }

            self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name)] = {
                'plotfn': plot_populations_RB_curve,
                'ncl': self.proc_data_dict['ncl'],
                'P0': self.proc_data_dict['P0'][val_name],
                'P1': self.proc_data_dict['P1'][val_name],
                'P2': self.proc_data_dict['P2'][val_name],
                'title': self.proc_data_dict['timestamp_string']+'\n' +
                'Population using rate equations ch{}'.format(val_name)
            }
        self.plot_dicts['logres_decision_bound'] = {
            'plotfn': plot_classifier_decission_boundary,
            'classifier': self.proc_data_dict['classifier'],
            'shots_0': (self.proc_data_dict['cal_pts_zero'][val_names[0]],
                        self.proc_data_dict['cal_pts_zero'][val_names[1]]),
            'shots_1': (self.proc_data_dict['cal_pts_one'][val_names[0]],
                        self.proc_data_dict['cal_pts_one'][val_names[1]]),
            'shots_2': (self.proc_data_dict['cal_pts_two'][val_names[0]],
                        self.proc_data_dict['cal_pts_two'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.proc_data_dict['value_units'][1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring'] +
            ' Decision boundary',
            'plotsize': (fs[0]*1.5, fs[1])}

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}


class RandomizedBenchmarking_TwoQubit_Analysis(
        RandomizedBenchmarking_SingleQubit_Analysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idxs: list =[0, 2],
                 ignore_f_cal_pts: bool=False
                 ):
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs,
            do_fitting=True)
        self.d1 = 4
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idxs = rates_ch_idxs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-7:2]  # 7 calibration points
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_x0'] = OrderedDict()
            self.raw_data_dict['cal_pts_x1'] = OrderedDict()
            self.raw_data_dict['cal_pts_x2'] = OrderedDict()
            self.raw_data_dict['cal_pts_0x'] = OrderedDict()
            self.raw_data_dict['cal_pts_1x'] = OrderedDict()
            self.raw_data_dict['cal_pts_2x'] = OrderedDict()

            self.raw_data_dict['measured_values_I'] = OrderedDict()
            self.raw_data_dict['measured_values_X'] = OrderedDict()

            for i, val_name in enumerate(a.value_names):
                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0) &
                                        (a.measured_values[2] == 0) &
                                        (a.measured_values[3] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*4)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')
                self.raw_data_dict['binned_vals'][val_name] = binned_yvals

                # 7 cal points:  [00, 01, 10, 11, 02, 20, 22]
                #      col_idx:  [-7, -6, -5, -4, -3, -2, -1]
                self.raw_data_dict['cal_pts_x0'][val_name] =\
                    binned_yvals[(-7, -5), :].flatten()
                self.raw_data_dict['cal_pts_x1'][val_name] =\
                    binned_yvals[(-6, -4), :].flatten()
                self.raw_data_dict['cal_pts_x2'][val_name] =\
                    binned_yvals[(-3, -1), :].flatten()

                self.raw_data_dict['cal_pts_0x'][val_name] =\
                    binned_yvals[(-7, -6), :].flatten()
                self.raw_data_dict['cal_pts_1x'][val_name] =\
                    binned_yvals[(-5, -4), :].flatten()
                self.raw_data_dict['cal_pts_2x'][val_name] =\
                    binned_yvals[(-2, -1), :].flatten()

                self.raw_data_dict['measured_values_I'][val_name] =\
                    binned_yvals[:-7:2, :]
                self.raw_data_dict['measured_values_X'][val_name] =\
                    binned_yvals[1:-7:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        for key in ['Vx0', 'V0x', 'Vx1', 'V1x', 'Vx2', 'V2x',
                    'SI', 'SX',
                    'Px0', 'P0x', 'Px1', 'P1x', 'Px2', 'P2x',
                    'M_inv_q0', 'M_inv_q1']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            for idx in ['x0', 'x1', 'x2', '0x', '1x', '2x']:
                self.proc_data_dict['V{}'.format(idx)][val_name] = \
                    np.nanmean(self.raw_data_dict['cal_pts_{}'.format(idx)]
                               [val_name])
            SI = np.nanmean(
                self.raw_data_dict['measured_values_I'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_X'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            Px0, Px1, Px2, M_inv_q0 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['Vx0'][val_name],
                self.proc_data_dict['Vx1'][val_name],
                self.proc_data_dict['Vx2'][val_name])
            P0x, P1x, P2x, M_inv_q1 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['V0x'][val_name],
                self.proc_data_dict['V1x'][val_name],
                self.proc_data_dict['V2x'][val_name])

            for key, val in [('Px0', Px0), ('Px1', Px1), ('Px2', Px2),
                             ('P0x', P0x), ('P1x', P1x), ('P2x', P2x),
                             ('M_inv_q0', M_inv_q0), ('M_inv_q1', M_inv_q1)]:
                self.proc_data_dict[key][val_name] = val

        if self.classification_method == 'rates':
            val_name_q0 = self.raw_data_dict['value_names'][self.rates_ch_idxs[0]]
            val_name_q1 = self.raw_data_dict['value_names'][self.rates_ch_idxs[1]]

            self.proc_data_dict['M0'] = (
                self.proc_data_dict['Px0'][val_name_q0] *
                self.proc_data_dict['P0x'][val_name_q1])

            self.proc_data_dict['X1'] = (
                1-self.proc_data_dict['Px2'][val_name_q0]
                - self.proc_data_dict['P2x'][val_name_q1])
        else:
            raise NotImplementedError()

    def prepare_plots(self):
        val_names = self.proc_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bins'],
                'yvals': np.nanmean(self.proc_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.proc_data_dict['binned_vals'][val_name], axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string'] +
                '\n'+self.proc_data_dict['measurementstring'],
            }
        fs = plt.rcParams['figure.figsize']

        # define figure and axes here to have custom layout
        self.figs['rb_populations_decay'], axs = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(fs[0]*1.5, fs[1]))
        self.figs['rb_populations_decay'].suptitle(
            self.proc_data_dict['timestamp_string']+'\n' +
            'Population using rate equations', y=1.05)
        self.figs['rb_populations_decay'].patch.set_alpha(0)
        self.axs['rb_pops_q0'] = axs[0]
        self.axs['rb_pops_q1'] = axs[1]

        val_name_q0 = val_names[self.rates_ch_idxs[0]]
        val_name_q1 = val_names[self.rates_ch_idxs[1]]
        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q0)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['Px0'][val_name_q0],
            'P1': self.proc_data_dict['Px1'][val_name_q0],
            'P2': self.proc_data_dict['Px2'][val_name_q0],
            'title': ' {}'.format(val_name_q0),
            'ax_id': 'rb_pops_q0'}

        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q1)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['P0x'][val_name_q1],
            'P1': self.proc_data_dict['P1x'][val_name_q1],
            'P2': self.proc_data_dict['P2x'][val_name_q1],
            'title': ' {}'.format(val_name_q1),
            'ax_id': 'rb_pops_q1'}

        self.plot_dicts['cal_points_hexbin_q0'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_x0'][val_names[0]],
                        self.proc_data_dict['cal_pts_x0'][val_names[1]]),
            'shots_1': (self.proc_data_dict['cal_pts_x1'][val_names[0]],
                        self.proc_data_dict['cal_pts_x1'][val_names[1]]),
            'shots_2': (self.proc_data_dict['cal_pts_x2'][val_names[0]],
                        self.proc_data_dict['cal_pts_x2'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.proc_data_dict['value_units'][1],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q0',
            'plotsize': (fs[0]*1.5, fs[1])
        }
        self.plot_dicts['cal_points_hexbin_q1'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_0x'][val_names[2]],
                        self.proc_data_dict['cal_pts_0x'][val_names[3]]),
            'shots_1': (self.proc_data_dict['cal_pts_1x'][val_names[2]],
                        self.proc_data_dict['cal_pts_1x'][val_names[3]]),
            'shots_2': (self.proc_data_dict['cal_pts_2x'][val_names[2]],
                        self.proc_data_dict['cal_pts_2x'][val_names[3]]),
            'xlabel': val_names[2],
            'xunit': self.proc_data_dict['value_units'][2],
            'ylabel': val_names[3],
            'yunit': self.proc_data_dict['value_units'][3],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q1',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}


class UnitarityBenchmarking_TwoQubit_Analysis(
        RandomizedBenchmarking_SingleQubit_Analysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None, auto=True, close_figs=True,
                 classification_method='rates', rates_ch_idxs: list =[0, 2],
                 ignore_f_cal_pts: bool=False, nseeds: int=None, **kwargs
                 ):
        """Analysis for unitarity benchmarking.

        This analysis is based on
        """
        if nseeds is None:
            raise TypeError('You must specify number of seeds!')
        self.nseeds = nseeds
        if options_dict is None:
            options_dict = dict()
        super(RandomizedBenchmarking_SingleQubit_Analysis, self).__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs,
            do_fitting=True, **kwargs)
        self.d1 = 4
        # used to determine how to determine 2nd excited state population
        self.classification_method = classification_method
        self.rates_ch_idxs = rates_ch_idxs
        self.ignore_f_cal_pts = ignore_f_cal_pts
        if auto:
            self.run_analysis()

    def extract_data(self):
        """Custom data extraction for Unitarity benchmarking.

        To determine the unitarity data is acquired in different bases.
        This method extracts that data and puts it in specific bins.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-7:10]  # 7 calibration points
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] = a.timestamp_string

            self.raw_data_dict['binned_vals'] = OrderedDict()
            self.raw_data_dict['cal_pts_x0'] = OrderedDict()
            self.raw_data_dict['cal_pts_x1'] = OrderedDict()
            self.raw_data_dict['cal_pts_x2'] = OrderedDict()
            self.raw_data_dict['cal_pts_0x'] = OrderedDict()
            self.raw_data_dict['cal_pts_1x'] = OrderedDict()
            self.raw_data_dict['cal_pts_2x'] = OrderedDict()

            self.raw_data_dict['measured_values_ZZ'] = OrderedDict()
            self.raw_data_dict['measured_values_XZ'] = OrderedDict()
            self.raw_data_dict['measured_values_YZ'] = OrderedDict()
            self.raw_data_dict['measured_values_ZX'] = OrderedDict()
            self.raw_data_dict['measured_values_XX'] = OrderedDict()
            self.raw_data_dict['measured_values_YX'] = OrderedDict()
            self.raw_data_dict['measured_values_ZY'] = OrderedDict()
            self.raw_data_dict['measured_values_XY'] = OrderedDict()
            self.raw_data_dict['measured_values_YY'] = OrderedDict()
            self.raw_data_dict['measured_values_mZmZ'] = OrderedDict()

            for i, val_name in enumerate(a.value_names):
                invalid_idxs = np.where((a.measured_values[0] == 0) &
                                        (a.measured_values[1] == 0) &
                                        (a.measured_values[2] == 0) &
                                        (a.measured_values[3] == 0))[0]
                a.measured_values[:, invalid_idxs] = \
                    np.array([[np.nan]*len(invalid_idxs)]*4)

                binned_yvals = np.reshape(
                    a.measured_values[i], (len(bins), -1), order='F')
                self.raw_data_dict['binned_vals'][val_name] = binned_yvals

                # 7 cal points:  [00, 01, 10, 11, 02, 20, 22]
                #      col_idx:  [-7, -6, -5, -4, -3, -2, -1]
                self.raw_data_dict['cal_pts_x0'][val_name] =\
                    binned_yvals[(-7, -5), :].flatten()
                self.raw_data_dict['cal_pts_x1'][val_name] =\
                    binned_yvals[(-6, -4), :].flatten()
                self.raw_data_dict['cal_pts_x2'][val_name] =\
                    binned_yvals[(-3, -1), :].flatten()

                self.raw_data_dict['cal_pts_0x'][val_name] =\
                    binned_yvals[(-7, -6), :].flatten()
                self.raw_data_dict['cal_pts_1x'][val_name] =\
                    binned_yvals[(-5, -4), :].flatten()
                self.raw_data_dict['cal_pts_2x'][val_name] =\
                    binned_yvals[(-2, -1), :].flatten()

                self.raw_data_dict['measured_values_ZZ'][val_name] =\
                    binned_yvals[0:-7:10, :]
                self.raw_data_dict['measured_values_XZ'][val_name] =\
                    binned_yvals[1:-7:10, :]
                self.raw_data_dict['measured_values_YZ'][val_name] =\
                    binned_yvals[2:-7:10, :]
                self.raw_data_dict['measured_values_ZX'][val_name] =\
                    binned_yvals[3:-7:10, :]
                self.raw_data_dict['measured_values_XX'][val_name] =\
                    binned_yvals[4:-7:10, :]
                self.raw_data_dict['measured_values_YX'][val_name] =\
                    binned_yvals[5:-7:10, :]
                self.raw_data_dict['measured_values_ZY'][val_name] =\
                    binned_yvals[6:-7:10, :]
                self.raw_data_dict['measured_values_XY'][val_name] =\
                    binned_yvals[7:-7:10, :]
                self.raw_data_dict['measured_values_YY'][val_name] =\
                    binned_yvals[8:-7:10, :]
                self.raw_data_dict['measured_values_mZmZ'][val_name] =\
                    binned_yvals[9:-7:10, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()  # closes data file

    def process_data(self):
        """Averages shot data and calculates unitarity from raw_data_dict.

        Note: this doe not correct the outcomes for leakage.



        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)

        keys = ['Vx0', 'V0x', 'Vx1', 'V1x', 'Vx2', 'V2x',
                'SI', 'SX',
                'Px0', 'P0x', 'Px1', 'P1x', 'Px2', 'P2x',
                'M_inv_q0', 'M_inv_q1']
        keys += ['XX', 'XY', 'XZ',
                 'YX', 'YY', 'YZ',
                 'ZX', 'ZY', 'ZZ',
                 'XX_sq', 'XY_sq', 'XZ_sq',
                 'YX_sq', 'YY_sq', 'YZ_sq',
                 'ZX_sq', 'ZY_sq', 'ZZ_sq',
                 'unitarity_shots', 'unitarity']
        keys += ['XX_q0', 'XY_q0', 'XZ_q0',
                 'YX_q0', 'YY_q0', 'YZ_q0',
                 'ZX_q0', 'ZY_q0', 'ZZ_q0']
        keys += ['XX_q1', 'XY_q1', 'XZ_q1',
                 'YX_q1', 'YY_q1', 'YZ_q1',
                 'ZX_q1', 'ZY_q1', 'ZZ_q1']
        for key in keys:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            for idx in ['x0', 'x1', 'x2', '0x', '1x', '2x']:
                self.proc_data_dict['V{}'.format(idx)][val_name] = \
                    np.nanmean(self.raw_data_dict['cal_pts_{}'.format(idx)]
                               [val_name])
            SI = np.nanmean(
                self.raw_data_dict['measured_values_ZZ'][val_name], axis=1)
            SX = np.nanmean(
                self.raw_data_dict['measured_values_mZmZ'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            Px0, Px1, Px2, M_inv_q0 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['Vx0'][val_name],
                self.proc_data_dict['Vx1'][val_name],
                self.proc_data_dict['Vx2'][val_name])
            P0x, P1x, P2x, M_inv_q1 = populations_using_rate_equations(
                SI, SX, self.proc_data_dict['V0x'][val_name],
                self.proc_data_dict['V1x'][val_name],
                self.proc_data_dict['V2x'][val_name])

            for key, val in [('Px0', Px0), ('Px1', Px1), ('Px2', Px2),
                             ('P0x', P0x), ('P1x', P1x), ('P2x', P2x),
                             ('M_inv_q0', M_inv_q0), ('M_inv_q1', M_inv_q1)]:
                self.proc_data_dict[key][val_name] = val

            for key in ['XX', 'XY', 'XZ',
                        'YX', 'YY', 'YZ',
                        'ZX', 'ZY', 'ZZ']:
                Vmeas = self.raw_data_dict['measured_values_'+key][val_name]
                Px2 = self.proc_data_dict['Px2'][val_name]
                V0 = self.proc_data_dict['Vx0'][val_name]
                V1 = self.proc_data_dict['Vx1'][val_name]
                V2 = self.proc_data_dict['Vx2'][val_name]
                val = Vmeas+0  # - (Px2*V2 - (1-Px2)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(
                    val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key+'_q0'][val_name] = val*2-1

                P2x = self.proc_data_dict['P2x'][val_name]
                V0 = self.proc_data_dict['V0x'][val_name]
                V1 = self.proc_data_dict['V1x'][val_name]

                # Leakage is ignored in this analysis.
                # V2 = self.proc_data_dict['V2x'][val_name]
                val = Vmeas + 0  # - (P2x*V2 - (1-P2x)*V1)[:,None]
                val -= V1
                val /= V0 - V1
                val = np.mean(np.reshape(
                    val, (val.shape[0], self.nseeds, -1)), axis=2)
                self.proc_data_dict[key+'_q1'][val_name] = val*2-1

        if self.classification_method == 'rates':
            val_name_q0 = self.raw_data_dict['value_names'][self.rates_ch_idxs[0]]
            val_name_q1 = self.raw_data_dict['value_names'][self.rates_ch_idxs[1]]

            self.proc_data_dict['M0'] = (
                self.proc_data_dict['Px0'][val_name_q0] *
                self.proc_data_dict['P0x'][val_name_q1])

            self.proc_data_dict['X1'] = (
                1-self.proc_data_dict['Px2'][val_name_q0]
                - self.proc_data_dict['P2x'][val_name_q1])

            # The unitarity is calculated here.
            self.proc_data_dict['unitarity_shots'] = \
                self.proc_data_dict['ZZ_q0'][val_name_q0]*0

            # Unitarity according to Eq. (10) Wallman et al. New J. Phys. 2015
            # Pj = d/(d-1)*|n(rho_j)|^2
            # Note that the dimensionality prefix is ignored here as it
            # should drop out in the fits.
            for key in ['XX', 'XY', 'XZ',
                        'YX', 'YY', 'YZ',
                        'ZX', 'ZY', 'ZZ']:
                self.proc_data_dict[key] = (
                    self.proc_data_dict[key+'_q0'][val_name_q0]
                    * self.proc_data_dict[key+'_q1'][val_name_q1])
                self.proc_data_dict[key+'_sq'] = self.proc_data_dict[key]**2

                self.proc_data_dict['unitarity_shots'] += \
                    self.proc_data_dict[key+'_sq']

            self.proc_data_dict['unitarity'] = np.mean(
                self.proc_data_dict['unitarity_shots'], axis=1)
        else:
            raise NotImplementedError()

    def run_fitting(self):
        super().run_fitting()
        self.fit_res['unitarity_decay'] = self.fit_unitarity_decay()

        unitarity_dec = self.fit_res['unitarity_decay'].params

        text_msg = 'Summary: \n'
        text_msg += format_value_string('Unitarity\n' +
                                        r'$u$', unitarity_dec['u'], '\n')
        text_msg += format_value_string(
            'Error due to\nincoherent mechanisms\n'+r'$\epsilon$',
            unitarity_dec['eps'])

        self.proc_data_dict['unitarity_msg'] = text_msg

    def fit_unitarity_decay(self):
        """Fits the data using the unitarity model."""
        fit_mod_unitarity = lmfit.Model(unitarity_decay, independent_vars='m')
        fit_mod_unitarity.set_param_hint(
            'A', value=.1, min=0, max=1, vary=True)
        fit_mod_unitarity.set_param_hint(
            'B', value=.8, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint(
            'u', value=.9, min=0, max=1, vary=True)

        fit_mod_unitarity.set_param_hint('d1', value=self.d1, vary=False)
        # Error due to incoherent sources
        # Feng Phys. Rev. Lett. 117, 260501 (2016) eq. (4)
        fit_mod_unitarity.set_param_hint('eps', expr='((d1-1)/d1)*(1-u**0.5)')

        params = fit_mod_unitarity.make_params()
        fit_mod_unitarity = fit_mod_unitarity.fit(
            data=self.proc_data_dict['unitarity'],
            m=self.proc_data_dict['ncl'], params=params)

        return fit_mod_unitarity

    def prepare_plots(self):
        val_names = self.proc_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['bins'],
                'yvals': np.nanmean(
                    self.proc_data_dict['binned_vals'][val_name], axis=1),
                'yerr':  sem(self.proc_data_dict['binned_vals'][val_name],
                             axis=1),
                'xlabel': 'Number of Cliffords',
                'xunit': '#',
                'ylabel': val_name,
                'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string'] +
                '\n'+self.proc_data_dict['measurementstring'],
            }
        fs = plt.rcParams['figure.figsize']

        # define figure and axes here to have custom layout
        self.figs['rb_populations_decay'], axs = plt.subplots(
            ncols=2, sharex=True, sharey=True, figsize=(fs[0]*1.5, fs[1]))
        self.figs['rb_populations_decay'].suptitle(
            self.proc_data_dict['timestamp_string']+'\n' +
            'Population using rate equations', y=1.05)
        self.figs['rb_populations_decay'].patch.set_alpha(0)
        self.axs['rb_pops_q0'] = axs[0]
        self.axs['rb_pops_q1'] = axs[1]

        val_name_q0 = val_names[self.rates_ch_idxs[0]]
        val_name_q1 = val_names[self.rates_ch_idxs[1]]
        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q0)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['Px0'][val_name_q0],
            'P1': self.proc_data_dict['Px1'][val_name_q0],
            'P2': self.proc_data_dict['Px2'][val_name_q0],
            'title': ' {}'.format(val_name_q0),
            'ax_id': 'rb_pops_q0'}

        self.plot_dicts['rb_rate_eq_pops_{}'.format(val_name_q1)] = {
            'plotfn': plot_populations_RB_curve,
            'ncl': self.proc_data_dict['ncl'],
            'P0': self.proc_data_dict['P0x'][val_name_q1],
            'P1': self.proc_data_dict['P1x'][val_name_q1],
            'P2': self.proc_data_dict['P2x'][val_name_q1],
            'title': ' {}'.format(val_name_q1),
            'ax_id': 'rb_pops_q1'}

        self.plot_dicts['cal_points_hexbin_q0'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_x0'][val_names[0]],
                        self.proc_data_dict['cal_pts_x0'][val_names[1]]),
            'shots_1': (self.proc_data_dict['cal_pts_x1'][val_names[0]],
                        self.proc_data_dict['cal_pts_x1'][val_names[1]]),
            'shots_2': (self.proc_data_dict['cal_pts_x2'][val_names[0]],
                        self.proc_data_dict['cal_pts_x2'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.proc_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.proc_data_dict['value_units'][1],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q0',
            'plotsize': (fs[0]*1.5, fs[1])
        }
        self.plot_dicts['cal_points_hexbin_q1'] = {
            'plotfn': plot_cal_points_hexbin,
            'shots_0': (self.proc_data_dict['cal_pts_0x'][val_names[2]],
                        self.proc_data_dict['cal_pts_0x'][val_names[3]]),
            'shots_1': (self.proc_data_dict['cal_pts_1x'][val_names[2]],
                        self.proc_data_dict['cal_pts_1x'][val_names[3]]),
            'shots_2': (self.proc_data_dict['cal_pts_2x'][val_names[2]],
                        self.proc_data_dict['cal_pts_2x'][val_names[3]]),
            'xlabel': val_names[2],
            'xunit': self.proc_data_dict['value_units'][2],
            'ylabel': val_names[3],
            'yunit': self.proc_data_dict['value_units'][3],
            'common_clims': False,
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'] +
            ' hexbin plot q1',
            'plotsize': (fs[0]*1.5, fs[1])
        }

        # define figure and axes here to have custom layout
        self.figs['main_rb_decay'], axs = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={'height_ratios': (2, 1)})
        self.figs['main_rb_decay'].patch.set_alpha(0)
        self.axs['main_rb_decay'] = axs[0]
        self.axs['leak_decay'] = axs[1]
        self.plot_dicts['main_rb_decay'] = {
            'plotfn': plot_rb_decay_woods_gambetta,
            'ncl': self.proc_data_dict['ncl'],
            'M0': self.proc_data_dict['M0'],
            'X1': self.proc_data_dict['X1'],
            'ax1': axs[1],
            'title': self.proc_data_dict['timestamp_string']+'\n' +
            self.proc_data_dict['measurementstring']}

        self.plot_dicts['fit_leak'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'leak_decay',
            'fit_res': self.fit_res['leakage_decay'],
            'setlabel': 'Leakage fit',
            'do_legend': True,
            'color': 'C2',
        }
        self.plot_dicts['fit_rb_simple'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay_simple'],
            'setlabel': 'Simple RB fit',
            'do_legend': True,
        }
        self.plot_dicts['fit_rb'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'main_rb_decay',
            'fit_res': self.fit_res['rb_decay'],
            'setlabel': 'Full RB fit',
            'do_legend': True,
            'color': 'C2',
        }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['rb_msg'],
            'xpos': 1.05, 'ypos': .6, 'ax_id': 'main_rb_decay',
            'horizontalalignment': 'left'}

        self.plot_dicts['correlated_readouts'] = {
            'plotfn': plot_unitarity_shots,
            'ncl': self.proc_data_dict['ncl'],
            'unitarity_shots': self.proc_data_dict['unitarity_shots'],
            'xlabel': 'Number of Cliffords',
            'xunit': '#',
            'ylabel': 'Unitarity',
            'yunit': '',
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'],
        }

        self.figs['unitarity'] = plt.subplots(nrows=1)
        self.plot_dicts['unitarity'] = {
            'plotfn': plot_unitarity,
            'ax_id': 'unitarity',
            'ncl': self.proc_data_dict['ncl'],
            'P': self.proc_data_dict['unitarity'],
            'xlabel': 'Number of Cliffords',
            'xunit': '#',
            'ylabel': 'Unitarity',
            'yunit': 'frac',
            'title': self.proc_data_dict['timestamp_string'] +
            '\n'+self.proc_data_dict['measurementstring'],
        }
        self.plot_dicts['fit_unitarity'] = {
            'plotfn': self.plot_fit,
            'ax_id': 'unitarity',
            'fit_res': self.fit_res['unitarity_decay'],
            'setlabel': 'Simple unitarity fit',
            'do_legend': True,
        }
        self.plot_dicts['unitarity_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['unitarity_msg'],
            'xpos': 0.6, 'ypos': .8, 'ax_id': 'unitarity',
            'horizontalalignment': 'left'}


def plot_cal_points_hexbin(shots_0,
                           shots_1,
                           shots_2,
                           xlabel: str, xunit: str,
                           ylabel: str, yunit: str,
                           title: str,
                           ax,
                           common_clims: bool=True,
                           **kw):
    # Choose colormap
    alpha_cmaps = []
    for cmap in [pl.cm.Blues, pl.cm.Reds, pl.cm.Greens]:
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        alpha_cmaps.append(my_cmap)

    f = plt.gcf()
    hb2 = ax.hexbin(x=shots_2[0], y=shots_2[1], cmap=alpha_cmaps[2])
    cb = f.colorbar(hb2, ax=ax)
    cb.set_label(r'Counts $|2\rangle$')

    hb1 = ax.hexbin(x=shots_1[0], y=shots_1[1], cmap=alpha_cmaps[1])
    cb = f.colorbar(hb1, ax=ax)
    cb.set_label(r'Counts $|1\rangle$')

    hb0 = ax.hexbin(x=shots_0[0], y=shots_0[1], cmap=alpha_cmaps[0])
    cb = f.colorbar(hb0, ax=ax)
    cb.set_label(r'Counts $|0\rangle$')

    if common_clims:
        clims = hb0.get_clim(), hb1.get_clim(), hb2.get_clim()
        clim = np.min(clims), np.max(clims)
        for hb in hb0, hb1, hb2:
            hb.set_clim(clim)

    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)


def plot_raw_RB_curve(ncl, SI, SX, V0, V1, V2, title, ax,
                      xlabel, xunit, ylabel, yunit, **kw):
    ax.plot(ncl, SI, label='SI', marker='o')
    ax.plot(ncl, SX, label='SX', marker='o')
    ax.plot(ncl[-1]+.5, V0, label='V0', marker='d', c='C0')
    ax.plot(ncl[-1]+1.5, V1, label='V1', marker='d', c='C1')
    ax.plot(ncl[-1]+2.5, V2, label='V2', marker='d', c='C2')
    ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.legend()


def plot_populations_RB_curve(ncl, P0, P1, P2, title, ax, **kw):
    ax.axhline(.5, c='k', lw=.5, ls='--')
    ax.plot(ncl, P0, c='C0', label=r'P($|g\rangle$)', marker='v')
    ax.plot(ncl, P1, c='C3', label=r'P($|e\rangle$)', marker='^')
    ax.plot(ncl, P2, c='C2', label=r'P($|f\rangle$)', marker='d')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('Population')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def plot_unitarity_shots(ncl, unitarity_shots, title, ax=None, **kw):
    ax.axhline(.5, c='k', lw=.5, ls='--')

    ax.plot(ncl, unitarity_shots, '.')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('unitarity')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)


def plot_unitarity(ncl, P, title, ax=None, **kw):
    ax.plot(ncl, P, 'o')

    ax.set_xlabel('Number of Cliffords (#)')
    ax.set_ylabel('unitarity')
    ax.grid(axis='y')
    ax.legend()
    ax.set_ylim(-.05, 1.05)
    ax.set_title(title)


def populations_using_rate_equations(SI: np.array, SX: np.array,
                                     V0: float, V1: float, V2: float):
    """
    Calculate populations using reference voltages.

    Args:
        SI (array): signal value for signal with I (Identity) added
        SX (array): signal value for signal with X (π-pulse) added
        V0 (float):
        V1 (float):
        V2 (float):
    returns:
        P0 (array): population of the |0> state
        P1 (array): population of the |1> state
        P2 (array): population of the |2> state
        M_inv (2D array) :  Matrix inverse to find populations

    Based on equation (S1) from Asaad & Dickel et al. npj Quant. Info. (2016)

    To quantify leakage, we monitor the populations Pi of the three lowest
    energy states (i ∈ {0, 1, 2}) and calculate the average
    values <Pi>. To do this, we calibrate the average signal levels Vi for
    the transmons in level i, and perform each measurement twice, the second
    time with an added final π pulse on the 0–1 transition. This final π
    pulse swaps P0 and P1, leaving P2 unaffected. Under the assumption that
    higher levels are unpopulated (P0 +P1 +P2 = 1),

     [V0 −V2,   V1 −V2] [P0]  = [S −V2]
     [V1 −V2,   V0 −V2] [P1]  = [S' −V2]

    where S (S') is the measured signal level without (with) final π pulse.
    The populations are extracted by matrix inversion.
    """
    M = np.array([[V0-V2, V1-V2], [V1-V2, V0-V2]])
    M_inv = np.linalg.inv(M)

    P0 = np.zeros(len(SI))
    P1 = np.zeros(len(SX))
    for i, (sI, sX) in enumerate(zip(SI, SX)):
        p0, p1 = np.dot(np.array([sI-V2, sX-V2]), M_inv)
        p0, p1 = np.dot(M_inv, np.array([sI-V2, sX-V2]))
        P0[i] = p0
        P1[i] = p1

    P2 = 1 - P0 - P1

    return P0, P1, P2, M_inv


def logisticreg_classifier_machinelearning(shots_0, shots_1, shots_2):
    """
    """
    # reshaping of the entries in proc_data_dict
    shots_0 = np.array(list(
        zip(list(shots_0.values())[0],
            list(shots_0.values())[1])))

    shots_1 = np.array(list(
        zip(list(shots_1.values())[0],
            list(shots_1.values())[1])))
    shots_2 = np.array(list(
        zip(list(shots_2.values())[0],
            list(shots_2.values())[1])))

    shots_0 = shots_0[~np.isnan(shots_0[:, 0])]
    shots_1 = shots_1[~np.isnan(shots_1[:, 0])]
    shots_2 = shots_2[~np.isnan(shots_2[:, 0])]

    X = np.concatenate([shots_0, shots_1, shots_2])
    Y = np.concatenate([0*np.ones(shots_0.shape[0]),
                        1*np.ones(shots_1.shape[0]),
                        2*np.ones(shots_2.shape[0])])

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    return logreg


def plot_classifier_decission_boundary(shots_0, shots_1, shots_2,
                                       classifier,
                                       xlabel: str, xunit: str,
                                       ylabel: str, yunit: str,
                                       title: str, ax, **kw):
    """
    Plot decision boundary on top of the hexbin plot of the training dataset.
    """
    grid_points = 200

    x_min = np.nanmin([shots_0[0], shots_1[0], shots_2[0]])
    x_max = np.nanmax([shots_0[0], shots_1[0], shots_2[0]])
    y_min = np.nanmin([shots_0[1], shots_1[1], shots_2[1]])
    y_max = np.nanmax([shots_0[1], shots_1[1], shots_2[1]])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                         np.linspace(y_min, y_max, grid_points))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plot_cal_points_hexbin(shots_0=shots_0,
                           shots_1=shots_1,
                           shots_2=shots_2,
                           xlabel=xlabel, xunit=xunit,
                           ylabel=ylabel, yunit=yunit,
                           title=title, ax=ax)
    ax.pcolormesh(xx, yy, Z,
                  cmap=c.ListedColormap(['C0', 'C3', 'C2']),
                  alpha=.2)


def plot_rb_decay_woods_gambetta(ncl, M0, X1, ax, ax1, title='', **kw):
    ax.plot(ncl, M0, marker='o', linestyle='')
    ax1.plot(ncl, X1, marker='d', linestyle='')
    ax.grid(axis='y')
    ax1.grid(axis='y')
    ax.set_ylim(-.05, 1.05)
    ax1.set_ylim(min(min(.97*X1), .92), 1.01)
    ax.set_ylabel(r'$M_0$ probability')
    ax1.set_ylabel(r'$X_1$ population')
    ax1.set_xlabel('Number of Cliffords')
    ax.set_title(title)


def leak_decay(A, B, lambda_1, m):
    """
    Eq. (9) of Wood Gambetta 2018.

        A ~= L2/ (L1+L2)
        B ~= L1/ (L1+L2) + eps_m
        lambda_1 = 1 - L1 - L2

    """
    return A + B*lambda_1**m


def full_rb_decay(A, B, C, lambda_1, lambda_2, m):
    """Eq. (15) of Wood Gambetta 2018."""
    return A + B*lambda_1**m+C*lambda_2**m


def unitarity_decay(A, B, u, m):
    """Eq. (8) of Wallman et al. New J. Phys. 2015."""
    return A + B*u**m


