'''
Collection of classes to analyse Quantum efficiency measurements.
General procedure is:
 - find factor 'a' of the quadratic scaling of the Single-Shot-Readout as a function of scaling_amp (see SSROAnalysis class)
 - find sigma from the Gaussian fit of the Ramsey as a function of scaling_amp (see RamseyAnalysis class)
 - Calculate eta = a * sigma**2 / 2 (see QuantumEfficiencyAnalysis class)
For details, see https://arxiv.org/abs/1711.05336

Lastly, the QuantumEfficiencyAnalysisTWPA class allows for analysing the efficiency
as a function of TWPA power and frequency.

Hacked together by Rene Vollmer
'''

import datetime
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar

import numpy as np
import lmfit
import os

from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict

import copy


class QuantumEfficiencyAnalysisTWPA(ba.BaseDataAnalysis):
    '''
    Analyses Quantum efficiency measurements as a function of TWPA Pump frequency and power.
    '''

    def __init__(self, t_start: str = None, t_stop: str = None, label_ramsey: str = '_Ramsey',
                 label_ssro: str = '_SSRO', label: str = '', options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 twpa_pump_freq_key: str = 'Instrument settings.TWPA_Pump.frequency',
                 twpa_pump_power_key: str = 'Instrument settings.TWPA_Pump.power'):
        '''

        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param options_dict: Available options are the ones from the base_analysis and:
                                - individual_plots : plot all the individual fits?
                                - cmap : colormap for 2D plots
                                - plotsize : plotsize for 2D plots
                                - (todo)
        :param auto: Execute all steps automatically
        :param close_figs: Close the figure (do not display)
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param label_ramsey: the label that was used to name the ramsey measurements
        :param label_ssro: the label that was used to name the SSRO measurements
        :param label: (Optional) common label that was used to name all measurements
        :param twpa_pump_freq_key: key for the TWPA Pump Frequency, e.g. 'Instrument settings.TWPA_Pump.frequency'
        :param twpa_pump_power_key: key for the TWPA Pump Power, e.g. 'Instrument settings.TWPA_Pump.power'
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)
        self.label_ramsey = label_ramsey
        self.label_ssro = label_ssro

        self.params_dict = {'TWPA_freq': twpa_pump_freq_key,
                            'TWPA_power': twpa_pump_power_key}

        self.numeric_params = ['TWPA_freq', 'TWPA_power']

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        # Sort data by frequencies and power
        self.proc_data_dict = {}
        twpa_freqs_unsorted = np.array(self.raw_data_dict['TWPA_freq'], dtype=float)
        twpa_freqs = np.unique(twpa_freqs_unsorted)
        twpa_freqs.sort()
        twpa_powers_unsorted = np.array(self.raw_data_dict['TWPA_power'], dtype=float)
        twpa_powers = np.unique(twpa_powers_unsorted)
        twpa_powers.sort()

        self.proc_data_dict['TWPA_freqs'] = twpa_freqs
        self.proc_data_dict['TWPA_powers'] = twpa_powers
        if self.verbose:
            print('Found %d twpa freqs and %d amplitudes' % (len(twpa_freqs), len(twpa_powers)))
            print(twpa_freqs, twpa_powers)

        dates = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        #date_limits = np.array([[(None, None)] * len(twpa_powers)] * len(twpa_freqs))
        datetimes = np.array(self.raw_data_dict['datetime'], dtype=datetime.datetime)
        for i, twpa_freq in enumerate(twpa_freqs):
            freq_indices = np.where(twpa_freqs_unsorted == twpa_freq)
            for j, twpa_power in enumerate(twpa_powers):
                power_indices = np.where(twpa_powers_unsorted == twpa_power)
                indices = np.array(np.intersect1d(freq_indices, power_indices), dtype=int)
                dts = datetimes[indices]
                dates[i, j] = dts
                #date_limits[i, j][0] = np.min(dts)
                #date_limits[i, j][1] = np.max(dts)

        self.proc_data_dict['sorted_datetimes'] = dates
        #self.proc_data_dict['sorted_date_limits'] = date_limits
    
    def process_data(self):
        twpa_freqs = self.proc_data_dict['TWPA_freqs']
        twpa_powers = self.proc_data_dict['TWPA_powers']
        dates = self.proc_data_dict['sorted_datetimes']
        #date_limits = self.proc_data_dict['sorted_date_limits']

        eta = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_eta = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        sigma = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_sigma = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        a = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_a = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)

        objects = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=QuantumEfficiencyAnalysis)
        d = copy.deepcopy(self.options_dict)
        d['save_figs'] = False
        for i, freq in enumerate(twpa_freqs):
            for j, power in enumerate(twpa_powers):
                t_start = [d.strftime("%Y%m%d_%H%M%S") for d in dates[i,j]] #date_limits[i, j][0]
                t_stop = None #date_limits[i, j][1]
                #print(t_start, t_stop)
                qea = QuantumEfficiencyAnalysis(t_start=t_start, t_stop=t_stop, label_ramsey=self.label_ramsey,
                                                label_ssro=self.label_ssro, options_dict=d, auto=False,
                                                extract_only=True)
                qea.run_analysis()

                a[i, j] = qea.fit_dicts['a']
                u_a[i, j] = qea.fit_dicts['a_std']

                sigma[i, j] = qea.fit_dicts['sigma']
                u_sigma[i, j] = qea.fit_dicts['sigma_std']

                eta[i, j] = qea.fit_dicts['eta']
                u_eta[i, j] = qea.fit_dicts['u_eta']

                objects[i,j] = qea  # Free up memory

        self.proc_data_dict['analysis_objects'] = objects

        self.proc_data_dict['as'] = a
        self.proc_data_dict['as_std'] = u_a
        self.proc_data_dict['sigmas'] = sigma
        self.proc_data_dict['sigmas_std'] = u_sigma
        self.proc_data_dict['etas'] = eta
        self.proc_data_dict['etas_std'] = u_eta

        self.proc_data_dict['as'] = a
        self.proc_data_dict['as_std'] = u_a
        self.proc_data_dict['sigmas'] = sigma
        self.proc_data_dict['sigmas_std'] = u_sigma
        self.proc_data_dict['etas'] = eta
        self.proc_data_dict['etas_std'] = u_eta

    def prepare_plots(self):
        twpa_powers = self.proc_data_dict['TWPA_powers']
        twpa_freqs = self.proc_data_dict['TWPA_freqs']

        #Quantum Efficiency
        self.plot_dicts['quantum_eff'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas'].transpose() * 100,
            'zlabel': r'Quantum efficiency $\eta$ (%)',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['quantum_eff_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas_std'].transpose(),
            'zlabel': r'Quantum efficiency Deviation $\delta \eta$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        #SSRO Slope
        self.plot_dicts['ssro_slope'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['as'].transpose(),
            'zlabel': r'SSRO slope $a$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['ssro_slope_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['as_std'].transpose(),
            'zlabel': r'SSRO slope variance $\delta a$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        #Ramsey Gauss Width
        self.plot_dicts['ramsey_gauss_width'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['sigmas'].transpose(),
            'zlabel': r'Ramsey Gauss width $\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['ramsey_gauss_width_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['sigmas_std'].transpose(),
            'zlabel': r'Ramsey Gauss width variance $\delta\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }


        if self.options_dict.get('individual_plots', False):
            # todo: add 1D plot from QuantumEfficiencyAnalysis
            for i, twpa_freq in enumerate(twpa_freqs):
                for j, twpa_power in enumerate(twpa_powers):
                    pre = 'freq_%.3f-power_%.3f-'%(twpa_freq,twpa_power)
                    obj = self.proc_data_dict['analysis_objects'][i,j]
                    for k in ['amp_vs_Ramsey_coherence', 'amp_vs_Ramsey_fit',]:
                        self.plot_dicts[pre+k] = obj.ra.plot_dicts[k]
                        self.plot_dicts[pre+k]['ax_id'] = pre+'snr_analysis'
                    for k in ['amp_vs_SNR_fit', 'amp_vs_SNR_scatter',]:
                        self.plot_dicts[pre+k] = obj.ssro.plot_dicts[k]
                        self.plot_dicts[pre+k]['ax_id'] = pre+'snr_analysis'



class QuantumEfficiencyAnalysis(ba.BaseDataAnalysis):
    '''
    Analyses one set of Quantum efficiency measurements
    '''

    def __init__(self, t_start: str = None, t_stop: str = None, label_ramsey: str = '_Ramsey',
                 label_ssro: str = '_SSRO', options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        '''

        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param options_dict: Available options are the ones from the base_analysis and:
                                - individual_plots : plot all the individual fits?
                                - cmap : colormap for 2D plots
                                - plotsize : plotsize for 2D plots
                                - (todo)
        :param auto: Execute all steps automatically
        :param close_figs: Close the figure (do not display)
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param label_ramsey: the label that was used to name the ramsey measurements
        :param label_ssro: the label that was used to name the SSRO measurements
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )


        d = copy.deepcopy(self.options_dict)
        d['save_figs'] = False
        self.ra = RamseyAnalysis(t_start=t_start, t_stop=t_stop, label=label_ramsey, options_dict=d,
                                 auto=False, extract_only=True)
        self.ssro = SSROAnalysis(t_start=t_start, t_stop=t_stop, label=label_ssro, options_dict=d,
                                 auto=False, extract_only=True)

        if auto:
            self.run_analysis()


    def extract_data(self):
        self.ra.extract_data()
        self.ssro.extract_data()
        
        self.raw_data_dict = OrderedDict()
        youngest = max(np.max(self.ra.raw_data_dict['datetime']), np.max(self.ssro.raw_data_dict['datetime']))
        youngest += datetime.timedelta(seconds=1)
        self.raw_data_dict['datetime'] = [youngest]
        self.raw_data_dict['timestamps'] = [youngest.strftime("%Y%m%d_%H%M%S")]
        self.timestamps = [youngest]


        f = '%s_quantum_efficiency_analysis' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = self.options_dict.get('analysis_result_file',
                                                                          os.path.join(folder, f + '.hdf5'))

    def run_fitting(self):
        self.ra.run_analysis()
        self.ssro.run_analysis()

        self.fit_dicts = OrderedDict()
        self.fit_dicts['sigma'] = self.ra.fit_dicts['coherence_fit']['sigma']
        self.fit_dicts['sigma_std'] = self.ra.fit_dicts['coherence_fit']['sigma_std']
        # self.raw_data_dict['scale'] = self.ra.fit_dicts['coherence_fit']['scale']
        self.fit_dicts['a'] = self.ssro.fit_dicts['snr_fit']['a']
        self.fit_dicts['a_std'] = self.ssro.fit_dicts['snr_fit']['a_std']

        sigma = self.fit_dicts['sigma']
        u_sigma = self.fit_dicts['sigma_std']
        a = self.fit_dicts['a']
        u_a = self.fit_dicts['a_std']

        eta = a * sigma ** 2 / 2
        u_eta = (u_a / a + 2 * u_sigma / sigma) * eta

        if self.verbose:
            print('eta = %.4f +- %.4f'%(eta, u_eta))

        self.fit_dicts['eta'] = np.array(eta, dtype=float)
        self.fit_dicts['u_eta'] = np.array(u_eta, dtype=float)

        # todo: Reformat data for saving in hdf5 file
        # self.fit_res['etas'] = {
        #    {power: self.proc_data_dict['etas'][i, j] for j, power in self.raw_data_dict['TWPA_freq_ramesy']}
        #    for i, freq in self.raw_data_dict['TWPA_power_ramesy']
        # }
        # self.fit_res['u_etas'] = {
        #    {power: self.proc_data_dict['u_etas'][i, j] for j, power in self.raw_data_dict['TWPA_freq_ramesy']}
        #    for i, freq in self.raw_data_dict['TWPA_power_ramesy']
        # }

    def prepare_plots(self):
        #self.ra.plot_dicts['amp_vs_Ramsey_fit']
        #self.ra.plot_dicts['amp_vs_Ramsey_Phase']
        #self.ra.plot_dicts['amp_vs_Ramsey_coherence']

        #self.ssro.plot_dicts['amp_vs_SNR_fit']
        #self.ssro.plot_dicts['amp_vs_SNR_scatter']
        #self.ssro.plot_dicts['amp_vs_Fa']
        #self.ssro.plot_dicts['amp_vs_Fd']
        self.ra.prepare_plots()
        for d in self.ra.plot_dicts:
            self.plot_dicts[d] = self.ra.plot_dicts[d]
        self.ssro.prepare_plots()
        for d in self.ssro.plot_dicts:
            self.plot_dicts[d] = self.ssro.plot_dicts[d]

        for k in ['amp_vs_Ramsey_coherence', 'amp_vs_Ramsey_fit', 'amp_vs_SNR_fit', 'amp_vs_SNR_scatter']:
            self.plot_dicts[k]['ax_id'] = 'snr_analysis'
            self.plot_dicts[k]['ylabel'] = 'SNR, coherence'
            self.plot_dicts[k]['yunit'] = 'a.u.'
            self.plot_dicts[k]['title'] = r'$\eta = (%.4f \pm %.4f)$ %%'%(100*self.fit_dicts['eta'], 100*self.fit_dicts['u_eta'])

        self.plot_dicts['amp_vs_Ramsey_fit']['color'] = 'red'
        self.plot_dicts['amp_vs_Ramsey_coherence']['color'] = 'red'
        self.plot_dicts['amp_vs_SNR_fit']['color'] = 'blue'
        self.plot_dicts['amp_vs_SNR_scatter']['color'] = 'blue'

        self.plot_dicts['amp_vs_Ramsey_coherence']['marker'] = 'o'
        self.plot_dicts['amp_vs_SNR_scatter']['marker'] = 'o'

        self.plot_dicts['amp_vs_SNR_fit']['do_legend'] = True
        self.plot_dicts['amp_vs_Ramsey_fit']['do_legend'] = True



class RamseyAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_Ramsey',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only
                         )

        self.params_dict = {'scaling_amp': 'Instrument settings.RO_lutman.M_amp_R0',
                            'dephasing': 'Analysis.Fitted Params lin_trans w0.amplitude.value',
                            'phase': 'Analysis.Fitted Params lin_trans w0.phase.value',
                            }
        self.numeric_params = ['scaling_amp', 'dephasing', 'phase']

        if auto:
            self.run_analysis()

    def extract_data(self):
        #Load data
        super().extract_data()

        #Set output paths
        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)

        f = '%s_amp_sweep_ramsey' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = self.options_dict.get('analysis_result_file',
                                                                          os.path.join(folder, f + '.hdf5'))

    def process_data(self):
        #Remove None entries
        dephasing = self.raw_data_dict['dephasing']
        amps = self.raw_data_dict['scaling_amp']
        mask = np.intersect1d(np.where(dephasing != None), np.where(amps != None))
        if self.options_dict.get('fit_cutoff', True):
            mask2 = np.where(amps < self.options_dict.get('fit_cutoff', 0.1))
            mask = np.intersect1d(mask, mask2)

        self.proc_data_dict['scaling_amp'] = amps[mask]
        self.proc_data_dict['dephasing'] = dephasing[mask]
        self.proc_data_dict['phase'] = self.raw_data_dict['phase'][mask]
        # todo: factor 2?
        self.proc_data_dict['coherence'] = self.raw_data_dict['dephasing'] / 2

    def run_fitting(self):
        self.fit_res = {}
        coherence = self.proc_data_dict['coherence']
        scaling_amp = self.proc_data_dict['scaling_amp']

        def gaussian(x, sigma, scale):
            return scale * np.exp(-(x) ** 2 / (2 * sigma ** 2))

        gmodel = lmfit.models.Model(gaussian)
        gmodel.set_param_hint('sigma', value=0.07, min=-1, max=1)
        gmodel.set_param_hint('scale', value=0.9) # , min=0.1, max=100)
        para = gmodel.make_params()
        coherence_fit = gmodel.fit(coherence, x=scaling_amp, **para)
        self.fit_res['coherence_fit'] = coherence_fit
        self.fit_dicts['coherence_fit'] = OrderedDict()
        self.fit_dicts['coherence_fit']['sigma'] = coherence_fit.params['sigma'].value
        self.fit_dicts['coherence_fit']['sigma_std'] = coherence_fit.params['sigma'].stderr
        self.fit_dicts['coherence_fit']['scale'] = coherence_fit.params['scale'].value
        self.fit_dicts['coherence_fit']['scale_std'] = coherence_fit.params['scale'].stderr

    def prepare_plots(self):
        name = ''
        self.plot_dicts[name + 'amp_vs_Ramsey_fit'] = {
            'plotfn': self.plot_fit,
            'ax_id': name + 'amp_vs_Ramsey',
            'zorder': 5,
            'fit_res': self.fit_res['coherence_fit'],
            'xvals': self.proc_data_dict['scaling_amp'],
            'marker': '',
            'linestyle': '-',
            'ylabel': 'Coherence',
            'yunit': 'a.u.',
            'xlabel': 'CLEAR scaling amplitude',
            'xunit': 'V',
            'setlabel': 'ramsey coherence fit',
        }
        self.plot_dicts[name + 'amp_vs_Ramsey_coherence'] = {
            'plotfn': self.plot_line,
            'ax_id': name + 'amp_vs_Ramsey',
            'zorder': 0,
            'xvals': self.proc_data_dict['scaling_amp'],
            'yvals': self.proc_data_dict['coherence'],
            'marker': 'x',
            'linestyle': '',
            'setlabel': 'ramsey coherence data',
        }
        self.plot_dicts[name + 'amp_vs_Ramsey_Phase'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['scaling_amp'],
            'yvals': self.proc_data_dict['phase'],
            'marker': 'x',
            'linestyle': '',
            'ylabel': 'Phase',
            'yunit': 'deg.',
            'xlabel': 'CLEAR scaling amplitude',
            'xunit': 'V',
            'setlabel': 'ramsey phase data',
        }


class SSROAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_SSRO',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )

        self.params_dict = {'scaling_amp': 'Instrument settings.RO_lutman.M_amp_R0',
                            'SNR': 'Analysis.SSRO_Fidelity.SNR',
                            'F_a': 'Analysis.SSRO_Fidelity.F_a',
                            'F_d': 'Analysis.SSRO_Fidelity.F_d',
                            }
        self.numeric_params = ['scaling_amp', 'SNR', 'F_a', 'F_d']

        if auto:
            self.run_analysis()

    def extract_data(self):
        #Load data
        super().extract_data()

        #Set output paths
        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)
        f = '%s_amp_sweep_SNR_optimized' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = self.options_dict.get('analysis_result_file',
                                                                          os.path.join(folder, f + '.hdf5'))

    def process_data(self):

        #Remove None entries
        snr = self.raw_data_dict['SNR']
        amps = self.raw_data_dict['scaling_amp']
        mask = np.intersect1d(np.where(snr != None), np.where(amps != None))
        if self.options_dict.get('fit_cutoff', True):
            mask2 = np.where(amps < self.options_dict.get('fit_cutoff', 0.1))
            mask = np.intersect1d(mask, mask2)

        self.proc_data_dict['scaling_amp'] = amps[mask]
        self.proc_data_dict['SNR'] = snr[mask]
        self.proc_data_dict['F_a'] = self.raw_data_dict['F_a'][mask]
        self.proc_data_dict['F_d'] = self.raw_data_dict['F_d'][mask]


    def run_fitting(self):
        self.fit_res = {}
        SNR = self.proc_data_dict['SNR']
        amp = self.proc_data_dict['scaling_amp']

        def line(x, a):
            return np.sqrt(a) * x

        gmodel = lmfit.models.Model(line)
        snr_fit = gmodel.fit(SNR, x=amp, a=5)
        self.fit_res['snr_fit'] = snr_fit

        self.fit_dicts['snr_fit'] = OrderedDict()
        self.fit_dicts['snr_fit']['a'] = snr_fit.params['a'].value
        self.fit_dicts['snr_fit']['a_std'] = snr_fit.params['a'].stderr

    def prepare_plots(self):
        name = ''
        self.plot_dicts[name + 'amp_vs_SNR_fit'] = {
            'plotfn': self.plot_fit,
            'ax_id': name + 'amp_vs_Ramsey',
            'zorder': 5,
            'fit_res': self.fit_res['snr_fit'],
            'xvals': self.proc_data_dict['scaling_amp'],
            'marker': '',
            'linestyle': '-',
            'setlabel': 'SNR fit',
            'do_legend': True,
        }
        self.plot_dicts[name + 'amp_vs_SNR_scatter'] = {
            'plotfn': self.plot_line,
            'ax_id': name + 'amp_vs_Ramsey',
            'zorder': 0,
            'xvals': self.proc_data_dict['scaling_amp'],
            'xlabel': 'CLEAR scaling amplitude',
            'xunit': 'V',
            'yvals': self.proc_data_dict['SNR'],
            'ylabel': 'SNR',
            'yunit': '-',
            'marker': 'x',
            'linestyle': '',
            'setlabel': 'SNR data',
            'do_legend': True,
        }
        self.plot_dicts[name + 'amp_vs_Fa'] = {
            'plotfn': self.plot_line,
            'zorder': 0,
            'ax_id': name + 'amp_vs_F',
            'xvals': self.proc_data_dict['scaling_amp'],
            'yvals': self.proc_data_dict['F_a'],
            'marker': 'x',
            'linestyle': '',
            'setlabel': '$F_a$ data',
            'do_legend': True,
        }
        self.plot_dicts[name + 'amp_vs_Fd'] = {
            'plotfn': self.plot_line,
            'zorder': 1,
            'ax_id': name + 'amp_vs_F',
            'xvals': self.proc_data_dict['scaling_amp'],
            'yvals': self.proc_data_dict['F_d'],
            'marker': 'x',
            'linestyle': '',
            'ylabel': 'Fidelity',
            'yunit': '-',
            'xlabel': 'CLEAR scaling amplitude',
            'xunit': 'V',
            'setlabel': '$F_d$ data',
            'do_legend': True,
        }