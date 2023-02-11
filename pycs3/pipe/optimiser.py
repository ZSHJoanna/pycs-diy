"""
Module to optimise the parameter of the noise generation. We keep only the DIC optimiser, since the likelihood evaluation is very
expensive computationally, the MCMC and PSO optimiser are very un-efficient. They are now removed from this version.
"""
import copy
import logging
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import multiprocess
import numpy as np

import pycs3.gen.lc_func
import pycs3.gen.stat
import pycs3.sim.draw
import pycs3.sim.twk as twk
import pycs3.spl.topopt

logger = logging.getLogger(__name__)


class Optimiser(object):
    """
    Class to optimise the parameters of the generative noise model. This currently works only for the 'PS_from_residuals'
    generative noise model. For now, we removed the PSOOptimiser and MCMCOptimiser children class because they were
    they were very unefficient. We keep only the DicOptimiser children class.
    """

    def __init__(self, lcs, fit_vector, spline, attachml_function, attachml_param, knotstep=None,
                 savedirectory="./", recompute_spline=True, max_core=None, theta_init=None,
                 n_curve_stat=32, shotnoise=None, tweakml_type='PS_from_residuals', display=False, verbose=False,
                 tweakml_name='', correction_PS_residuals=True, tolerance=0.75, debug=False):

        """

        :param lcs: list of LightCurves
        :param fit_vector: list, target vector containing the target value of zruns and sigma for each LightCurve
        :param spline: Spline, intrinsic Spline corresponding to your lcs
        :param attachml_function: function, can be pycs3.gen.splml.addtolc() or pycs3.gen.polyml.addtolc()
        :param attachml_param: float, argument to be passed to the attachml_function. It is a float for the moment as atachml functions typically requires only one argument. This is typically the mlknostep of your microlensing model.
        :param knotstep: float, knotstep of the intrinsic spline
        :param savedirectory: string, name of the directory to save the output
        :param recompute_spline: boolean, this should stay to True unless you know what you are doing
        :param max_core: integer, number of cores used for the computation if using multhreading.
        :param theta_init: list, starting point of the optimisation
        :param n_curve_stat: float, number of trial curves to compute the statistics
        :param shotnoise: string, shotnoise type, should be left to None, unless you know what you are doing.
        :param tweakml_type: str, only 'PS_from_residuals' implemented so far
        :param display: boolean, to display the figure
        :param verbose: boolean, Verbosity
        :param tweakml_name: string, Name of the tweakml function, in case you try several type.
        :param correction_PS_residuals: boolean, to add an additionnal correction for the scaling of the Power Spectrum,should be set to True
        :param tolerance: float, tolerance in unit of sigma. I stop the optimisation if I found a set of parameters matching the data within the tolerance limit.
        :param debug: boolean, debug mode, not using multithreading
        """

        if len(fit_vector) != len(lcs):  # pragma: no cover
            raise RuntimeError("Your target vector and list of light curves must have the same size !")
        if recompute_spline is True and knotstep is None:  # pragma: no cover
            raise RuntimeError(" I can't recompute spline if you don't give me the knotstep ! ")
        if tweakml_type != 'PS_from_residuals':  # pragma: no cover
            raise NotImplementedError(
                "Other tweakml_type are not yet implemented in the optimiser, choose PS_from_residuals.")

        self.lcs = lcs
        self.ncurve = len(lcs)
        self.fit_vector = np.asarray(fit_vector)
        self.spline = spline
        self.attachml_function = attachml_function
        self.attachml_param = attachml_param
        if theta_init is None:
            theta_init = [[0.5] for i in range(self.ncurve)]
        if len(theta_init) != len(lcs): # pragma: no cover
            raise RuntimeError("Your init vector and list of light curves must have the same size !")
        self.theta_init = theta_init
        self.knotstep = knotstep
        self.savedirectory = savedirectory
        self.recompute_spline = recompute_spline
        self.success = False
        self.mean_zruns_mini = None  # mean zruns computed with the best parameters
        self.mean_sigma_mini = None  # mean sigma computed with the best parameters
        self.std_zruns_mini = None  # error on  sigma computed with the best parameters
        self.std_sigma_mini = None  # error of zruns and sigma computed with the best parameters
        self.chi2_mini = None
        self.rel_error_zruns_mini = None  # relative error in term of zruns
        self.rel_error_sigmas_mini = None  # relative error in term of sigmas
        self.best_param = None
        self.time_start = None
        self.time_stop = None

        if max_core is not None:
            self.max_core = max_core
        else:
            self.max_core = multiprocess.cpu_count()
            logger.info("You will run on %i cores." % self.max_core)

        if debug:
            self.para = False
        else:
            self.para = True
            logger.debug("Debug mode, I won't compute the mock curves in parallel.")

        self.n_curve_stat = n_curve_stat
        self.shotnoise = shotnoise
        self.shotnoisefrac = 1.0
        self.tweakml_type = tweakml_type
        self.tweakml_name = tweakml_name
        self.correction_PS_residuals = correction_PS_residuals  # boolean to set if you want to use the correction, True by default
        self.A_correction = np.ones(
            self.ncurve)  # this is the correction for the amplitude of the power spectrum of the risduals, this is use only for PS_from_residuals
        self.display = display
        self.verbose = verbose
        self.grid = None
        self.message = '\n'
        self.error_message = []
        self.tolerance = tolerance  # tolerance in unit of sigma for the fit
        self.timeshifts = [l.timeshift for l in self.lcs]
        self.magshifts = [l.magshift for l in self.lcs]

    def make_mocks_para(self, theta): # pragma: no cover #parralel computing, cannot be covered
        """
        Draw mock curves, optimise them, and compute the zrun and sigma statistics.  This is used in debug mode.
        It does the same than make_mocks but using multithreading.

        :param theta: list, containing the parameter of the generative noise model.
        :return: tuple (mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas)
        """
        stat = []
        zruns = []
        sigmas = []
        nruns = []

        pool = multiprocess.Pool(processes=self.max_core)

        job_args = [theta for j in range(self.n_curve_stat)]

        out = pool.map(self.fct_para, job_args)
        pool.close()
        pool.join()

        stat_out = np.asarray([x['stat'] for x in out if x['stat'] is not None])  # clean failed optimisation
        message_out = np.asarray([x['error'] for x in out if x['error'] is not None])  # clean failed optimisation
        self.error_message.append(message_out)
        zruns = np.asarray([[stat_out[i, j]['zruns'] for j in range(self.ncurve)] for i in range(len(stat_out))])
        sigmas = np.asarray([[stat_out[i, j]['std'] for j in range(self.ncurve)] for i in range(len(stat_out))])
        nruns = np.asarray([[stat_out[i, j]['nruns'] for j in range(self.ncurve)] for i in range(len(stat_out))])

        mean_zruns = []
        std_zruns = []
        mean_sigmas = []
        std_sigmas = []

        for i in range(self.ncurve):
            mean_zruns.append(np.mean(zruns[:, i]))
            std_zruns.append(np.std(zruns[:, i]))
            mean_sigmas.append(np.mean(sigmas[:, i]))
            std_sigmas.append(np.std(sigmas[:, i]))
            if self.verbose:
                logger.info('Curve %i :' % (i + 1))
                logger.info(f'Mean zruns (simu):  {np.mean(zruns[:, i])} +/-  {np.std(zruns[:, i])}')
                logger.info(f'Mean sigmas (simu): {np.mean(sigmas[:, i])} +/-  {np.std(sigmas[:, i])}')
                logger.info(f'Mean nruns (simu):  {np.mean(nruns[:, i])} +/-  {np.std(nruns[:, i])}')

        return mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas

    def compute_chi2(self, theta):
        """
        Likelihood function of the fit. Return chi2 metric between the current fit vector and the target vector.

        :param theta: list, containing the parameter of the generative noise model.
        :return:
        """

        chi2 = 0.0
        count = 0.0
        if self.n_curve_stat == 1:  # pragma: no cover
            raise RuntimeError(" I cannot compute statistics with one single curves !! Increase n_curve_stat.")

        if self.para:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks_para(theta)
        else:  # pragma: no cover #Used in debug mode
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks(theta)

        for i in range(self.ncurve):
            chi2 += (self.fit_vector[i][0] - mean_zruns[i]) ** 2 / std_zruns[i] ** 2
            chi2 += (self.fit_vector[i][1] - mean_sigmas[i]) ** 2 / (std_sigmas[i] ** 2)
            count += 1.0

        chi2 = chi2 / count
        return chi2, np.asarray(mean_zruns), np.asarray(mean_sigmas), np.asarray(std_zruns), np.asarray(std_sigmas)

    def fct_para(self, theta): # pragma: no cover
        """
        Auxilliary function to optimise the mock curves in parallel. See make_mocks_para().

        :param theta: list, containing the parameter of the generative noise model.

        :return: dcitionnary containing the stats and eventual error message
        """

        tweak_list = self.get_tweakml_list(theta)
        mocklc = pycs3.sim.draw.draw(self.lcs, self.spline,
                                     tweakml=tweak_list, shotnoise=self.shotnoise, scaletweakresi=False,
                                     shotnoisefrac=self.shotnoisefrac, keeptweakedml=False, keepshifts=False,
                                     keeporiginalml=False,
                                     inprint_fake_shifts=None)  # this return mock curve without ML

        pycs3.gen.lc_func.applyshifts(mocklc, self.timeshifts, [-np.median(lc.getmags()) for lc in mocklc])
        self.attachml_function(mocklc, self.attachml_param)  # adding the microlensing here ! Before the optimisation

        if self.recompute_spline:
            if self.knotstep is None:
                logger.error("You must give a knotstep to recompute the spline")
            try:
                spline_on_mock = pycs3.spl.topopt.opt_fine(mocklc, nit=5, knotstep=self.knotstep,
                                                           verbose=self.verbose, bokeps=self.knotstep / 3.0,
                                                           stabext=100)
                mockrls = pycs3.gen.stat.subtract(mocklc, spline_on_mock)
                stat = pycs3.gen.stat.mapresistats(mockrls)
            except Exception as e:
                logger.warning('Light curves could not be optimised for parameter :', theta)
                error_message = 'The following error occured : %s for parameters %s \n' % (e, str(theta))
                return {'stat': None, 'error': error_message}
            else:
                return {'stat': stat, 'error': None}
        else:
            mockrls = pycs3.gen.stat.subtract(mocklc, self.spline)
            stat = pycs3.gen.stat.mapresistats(mockrls)
            return {'stat': stat, 'error': None}

    def get_tweakml_list(self, theta):
        """
        Define the tweakml function for the corresponding fit vector

        :param theta: list, containing the parameter of the generative noise model.
        :return:
        """
        tweak_list = []

        if self.tweakml_type == 'PS_from_residuals':
            def tweakml_PS(lcs, spline, B, A_correction):
                return twk.tweakml_PS(lcs, spline, B, f_min=1 / 300.0, psplot=False, save_figure_folder=None,
                                      verbose=self.verbose, interpolation='linear', A_correction=A_correction)

            for i in range(self.ncurve):
                tweak_list.append(partial(tweakml_PS, B=theta[i][0], A_correction=self.A_correction[i]))
        else:  # pragma: no cover
            raise NotImplementedError('Other Tweakml_type than PS_from_residuals are not yet implemented.')

        return tweak_list

    def fct_para_aux(self, args): # pragma: no cover
        """
        Auxilliary function for parallel computing
        :param args:

        """
        kwargs = args[-1]
        args = args[0:-1]
        return self.fct_para(*args, **kwargs)

    def make_mocks(self, theta):
        """
        Draw mock curves, optimise them, and compute the zrun and sigma statistics.  This is used in debug mode.
        It does the same than make_mocks_para but serially.
        :param theta: list, containing the parameter of the generative noise model.

        :return: tuple (mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas)
        """

        mocklc = []
        mockrls = []
        stat = []
        zruns = []
        sigmas = []
        nruns = []
        lcscopies = [l.copy() for l in self.lcs]
        spline_copy = self.spline.copy()

        for i in range(self.n_curve_stat):
            tweak_list = self.get_tweakml_list(theta)
            mocklc.append(pycs3.sim.draw.draw(lcscopies, spline_copy,
                                              tweakml=tweak_list, shotnoise=self.shotnoise,
                                              shotnoisefrac=self.shotnoisefrac,
                                              keeptweakedml=False, keepshifts=False, keeporiginalml=False,
                                              scaletweakresi=False,
                                              inprint_fake_shifts=None))  # this will return mock curve WITHOUT microlensing !

            pycs3.gen.lc_func.applyshifts(mocklc[i], self.timeshifts, [-np.median(lc.getmags()) for lc in mocklc[i]])
            self.attachml_function(mocklc[i], self.attachml_param)  # adding the microlensing here

            if self.recompute_spline:
                if self.knotstep is None:  # pragma: no cover
                    raise RuntimeError("You must give a knotstep to recompute the spline")
                spline_on_mock = pycs3.spl.topopt.opt_fine(mocklc[i], nit=5, knotstep=self.knotstep,
                                                           verbose=self.verbose, bokeps=self.knotstep / 3.0,
                                                           stabext=100)  # TODO : maybe pass the optimisation function to the class in argument
                mockrls.append(pycs3.gen.stat.subtract(mocklc[i], spline_on_mock))
            else:
                mockrls.append(pycs3.gen.stat.subtract(mocklc[i], self.spline))

            if self.recompute_spline and self.display:
                pycs3.gen.lc_func.display(lcscopies, [spline_on_mock], showdelays=True)
                pycs3.gen.lc_func.display(mocklc[i], [spline_on_mock], showdelays=True)
                pycs3.gen.stat.plotresiduals([mockrls[i]])

            stat.append(pycs3.gen.stat.mapresistats(mockrls[i]))
            zruns.append([stat[i][j]['zruns'] for j in range(self.ncurve)])
            sigmas.append([stat[i][j]['std'] for j in range(self.ncurve)])
            nruns.append([stat[i][j]['nruns'] for j in range(self.ncurve)])

        zruns = np.asarray(zruns)
        sigmas = np.asarray(sigmas)
        nruns = np.asarray(nruns)
        mean_zruns = []
        std_zruns = []
        mean_sigmas = []
        std_sigmas = []
        for i in range(self.ncurve):
            mean_zruns.append(np.mean(zruns[:, i]))
            std_zruns.append(np.std(zruns[:, i]))
            mean_sigmas.append(np.mean(sigmas[:, i]))
            std_sigmas.append(np.std(sigmas[:, i]))
            if self.verbose:
                logger.info('Curve %s :' % self.lcs[i].object)
                logger.info(f'Mean zruns (simu):  {np.mean(zruns[:, i])} +/-  {np.std(zruns[:, i])}')
                logger.info(f'Mean sigmas (simu): {np.mean(sigmas[:, i])} +/-  {np.std(sigmas[:, i])}')
                logger.info(f'Mean nruns (simu):  {np.mean(nruns[:, i])} +/-  {np.std(nruns[:, i])}')

        return mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas

    def check_success(self):
        """
        Check if the optimiser found a set of parameter within the tolerance.

        :return: boolean

        """
        if any(self.rel_error_zruns_mini[i] is None for i in range(self.ncurve)):  # pragma: no cover
            raise RuntimeError("Error you should run analyse_plot_results() first !")
        else:
            if all(self.rel_error_zruns_mini[i] < self.tolerance for i in range(self.ncurve)) \
                    and all(self.rel_error_sigmas_mini[i] < self.tolerance for i in range(self.ncurve)):
                return True
            else:
                return False

    def report(self):
        """
        Write the optimisation report

        """
        if self.chain_list is None:  # pragma: no cover
            raise RuntimeError("You should run optimise() first ! I can't write the report")

        f = open(os.path.join(self.savedirectory, 'report_tweakml_optimisation.txt'), 'a')
        for i in range(self.ncurve):

            f.write('Lightcurve %s : \n' % self.lcs[i].object)
            f.write('\n')
            if self.rel_error_zruns_mini[i] < self.tolerance and self.rel_error_sigmas_mini[i] < self.tolerance:
                f.write('I succeeded in finding a set of parameters that match the '
                        'statistical properties of the real lightcurve within %2.2f sigma. \n' % self.tolerance)

            else:
                f.write('I did not succeed in finding a set of parameters that '
                        'match the statistical properties of the real lightcurve within %2.2f sigma. \n' % self.tolerance)

            f.write(self.message)
            f.write('Best parameters are : %s \n' % str(self.best_param[i]))
            if self.tweakml_type == 'PS_from_residuals':
                f.write('A correction for PS_from_residuals : %2.2f \n' % self.A_correction[i])
            f.write("Corresponding Chi2 : %2.2f \n" % self.chi2_mini)
            f.write("Target zruns, sigma : %2.6f, %2.6f \n" % (self.fit_vector[i, 0], self.fit_vector[i, 1]))
            f.write("At minimum zruns, sigma : %2.6f +/- %2.6f, %2.6f +/- %2.6f \n" % (
                self.mean_zruns_mini[i], self.std_zruns_mini[i],
                self.mean_sigma_mini[i], self.std_sigma_mini[i]))
            f.write("For minimum Chi2, we are standing at " + str(self.rel_error_zruns_mini[i]) + " sigma [zruns] \n")
            f.write("For minimum Chi2, we are standing at " + str(self.rel_error_sigmas_mini[i]) + " sigma [sigma] \n")
            f.write('------------------------------------------------\n')
            f.write('\n')
        f.write('Optimisation done in %4.4f seconds on %i cores' % ((self.time_stop - self.time_start), self.max_core))
        f.close()

        # Write the error report :
        g = open(os.path.join(self.savedirectory, 'errors_tweakml_optimisation.txt'), 'a')
        for mes in self.error_message:
            if not len(mes) == 0:
                g.write(str(mes))
        g.close()

    def reset_report(self):
        """
        Delete the optimisation report

        """
        if os.path.isfile(os.path.join(self.savedirectory, 'report_tweakml_optimisation.txt')):
            os.remove(os.path.join(self.savedirectory, 'report_tweakml_optimisation.txt'))

    def compute_set_A_correction(self, eval_pts):
        """
        This function compute the sigma obtained after optimisation in the middle of the grid and return the correction
         that will be used for the rest of the optimisation

        :param eval_pts: vector containing the generative noise models parameter
        :return: tuple (A_correction, mean_zruns, mean_sigmas, std_zruns, std_sigmas)
        """
        self.A_correction = [1.0 for i in range(self.ncurve)]  # reset the A correction

        if self.para:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks_para(eval_pts)
        else:  # pragma: no cover #Used in debug mode
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks(eval_pts)

        self.A_correction = self.fit_vector[:, 1] / mean_sigmas  # set the A correction
        return self.A_correction, mean_zruns, mean_sigmas, std_zruns, std_sigmas


class DicOptimiser(Optimiser):
    """
    Dichotomy search optimiser. It inherit from the Optimiser class.

    """

    def __init__(self, lcs, fit_vector, spline, attachml_function, attachml_param, knotstep=None,
                 savedirectory="./", recompute_spline=True, max_core=None, theta_init=None,
                 n_curve_stat=32, shotnoise=None, tweakml_type='PS_from_residuals', tweakml_name='',
                 display=False, verbose=False, step=0.1, correction_PS_residuals=True, max_iter=10, tolerance=0.75
                 , debug=False):

        Optimiser.__init__(self, lcs, fit_vector, spline, attachml_function, attachml_param,
                           knotstep=knotstep, savedirectory=savedirectory, recompute_spline=recompute_spline,
                           max_core=max_core, n_curve_stat=n_curve_stat, shotnoise=shotnoise, theta_init=theta_init,
                           tweakml_type=tweakml_type, tweakml_name=tweakml_name,
                           correction_PS_residuals=correction_PS_residuals,
                           verbose=verbose, display=display, tolerance=tolerance, debug=debug)

        self.chain_list = None
        self.step = [step for i in range(self.ncurve)]
        self.max_iter = max_iter
        self.iteration = 0
        self.turn_back = [0 for i in range(self.ncurve)]
        self.explored_param = []

    def optimise(self):
        """
        High-level function. Run the optimisation.

        """

        self.time_start = time.time()
        sigma = []
        zruns = []
        sigma_std = []
        zruns_std = []
        chi2 = []
        zruns_target = self.fit_vector[:, 0]
        sigma_target = self.fit_vector[:, 1]
        B = copy.deepcopy(self.theta_init)

        if self.correction_PS_residuals:
            self.A_correction, _, _, _, _ = self.compute_set_A_correction(B)
            logger.info(f"I will slightly correct the amplitude of the Power Spectrum by a factor : {np.array2string(np.asarray(self.A_correction))}")

        while True:
            self.iteration += 1
            logger.info("Iteration %i, B vector : " %self.iteration + np.array2string(np.asarray(B)))
            chi2_c, zruns_c, sigma_c, zruns_std_c, sigma_std_c = self.compute_chi2(B)

            chi2.append(chi2_c)
            sigma.append(sigma_c)
            zruns.append(zruns_c)
            sigma_std.append(sigma_std_c)
            zruns_std.append(zruns_std_c)
            self.explored_param.append(copy.deepcopy(B))

            self.rel_error_zruns_mini = np.abs(
                zruns_c - self.fit_vector[:, 0]) / zruns_std_c  # used to store the current relative error
            self.rel_error_sigmas_mini = np.abs(sigma_c - self.fit_vector[:, 1]) / sigma_std_c

            for i in range(self.ncurve):
                if self.step[i] > 0 and zruns_c[i] > zruns_target[i]:
                    self.turn_back[i] += 1
                    if self.iteration != 1:
                        self.step[i] = - self.step[i] / 2.0  # we go backward dividing the step by two
                    else:
                        self.step[i] = - self.step[
                            i]  # we do two step backward if the first iteration was already too high.

                elif self.step[i] < 0 and zruns_c[i] < zruns_target[i]:
                    self.turn_back[i] += 1
                    self.step[i] = - self.step[i] / 2.0  # we go backward dividing the step by two

                elif self.step[i] > 0.6:  # max step size
                    self.step[i] = 0.6

                elif B[i][0] <= 0.4 and self.step[i] <= -0.2:  # condition to reach 0.1 aymptotically

                    self.step = self.step / 2.0

                elif self.iteration % 3 == 0 and self.turn_back[i] == 0:
                    self.step[i] = self.step[
                                       i] * 2.0  # we double the step every 3 iterations if we didn't pass the optimum

            if self.check_if_stop():
                break

            for i in range(self.ncurve):
                B[i][0] += self.step[i]
                if B[i][0] <= 0.05:
                    B[i][0] = 0.05  # minimum for B
                if B[i][0] >= 4.0:
                    B[i][0] = 4.0  # maximum for B

            if self.iteration % 5 == 0:
                self.A_correction, _, _, _, _ = self.compute_set_A_correction(
                    B)  # recompute A correction every 5 iterations.
                logger.info(f"I will slightly correct the amplitude of the Power Spectrum by a factor : {np.array2string(np.asarray(self.A_correction))}")

        self.chain_list = [self.explored_param, chi2, zruns, sigma, zruns_std,
                           sigma_std]  # explored param has dimension(n_iter,ncurve,1)
        self.chi2_mini, self.best_param = chi2[-1], self.explored_param[
            -1]  # take the last iteration as the best estimate
        self.mean_zruns_mini = zruns[-1]
        self.std_zruns_mini = zruns_std[-1]
        self.mean_sigma_mini = sigma[-1]
        self.std_sigma_mini = sigma_std[-1]
        self.success = self.check_success()
        self.time_stop = time.time()
        return self.chain_list

    def check_if_stop(self):
        """
        Check if one of the stopping condition is met.

        :return: bool

        """
        if self.iteration >= self.max_iter:
            self.message = "I stopped because I reached the max number of iteration.\n"
            logger.info(self.message[:-2])
            return True
        if all(self.turn_back[i] > 4 for i in range(self.ncurve)):
            self.message = "I stopped because I passed four times the optimal value for all the curves.\n"
            logger.info(self.message[:-2])
            return True
        if self.check_success():
            self.message = "I stopped because I found a good set of parameters. \n"
            logger.info(self.message[:-2])
            return True
        else:
            return False

    def get_best_param(self):
        """
        Return the best fit parameters of the generative noise model. To be called after optimise()

        """
        if self.chain_list is None:  # pragma: no cover
            raise RuntimeError("I don't have the best parameters yet. You should run optimise() first !")
        else:
            ind_min = np.argmin(self.chain_list[1][:])
            self.chi2_mini = np.min(self.chain_list[1][:])
            return self.chi2_mini, self.chain_list[0][ind_min]

    def analyse_plot_results(self):
        """
        This is currently only producing the plot. You might want to do some other fancy operation here in the future.

        """
        self.plot_chain_grid_dic()

    def plot_chain_grid_dic(self):
        """
        Make the diagnostic plots

        """
        for i, l in enumerate(self.lcs):
            x_param = np.asarray(self.explored_param)[:, i]
            z_runs = np.asarray(self.chain_list[2])[:, i]
            z_runs_err = np.asarray(self.chain_list[4])[:, i]
            sigmas = np.asarray(self.chain_list[3])[:, i]
            sigmas_err = np.asarray(self.chain_list[5])[:, i]

            fig1 = plt.figure(1)
            plt.errorbar(x_param[:, 0], z_runs, yerr=z_runs_err, marker='o')
            plt.hlines(self.fit_vector[i, 0], np.min(x_param[:, 0]), np.max(x_param[:, 0]), colors='r',
                       linestyles='solid', label='target')
            plt.xlabel('B in unit of Nymquist frequency)')
            plt.ylabel('zruns')
            plt.legend()

            fig2 = plt.figure(2)
            plt.errorbar(x_param[:, 0], sigmas, yerr=sigmas_err, marker='o')
            plt.hlines(self.fit_vector[i, 1], np.min(x_param[:, 0]), np.max(x_param[:, 0]), colors='r',
                       linestyles='solid', label='target')
            plt.xlabel('B in unit of Nymquist frequency)')
            plt.ylabel('sigma')
            plt.legend()

            fig1.savefig(os.path.join(self.savedirectory, self.tweakml_name + '_zruns_' + l.object + '.png'))
            fig2.savefig(os.path.join(self.savedirectory, self.tweakml_name + '_std_' + l.object + '.png'))

            if self.display:  # pragma: no cover
                plt.show()
            plt.clf()
            plt.close('all')

        fig3 = plt.figure(3)
        x = np.arange(1, len(self.chain_list[1]) + 1, 1)
        plt.plot(x, self.chain_list[1])
        plt.xlabel('Iteration')
        plt.ylabel(r'$\chi^2$')
        fig3.savefig(os.path.join(self.savedirectory, self.tweakml_name + '_chi2.png'))


def get_fit_vector(lcs, spline):
    """
    Return the target vector containing the value of sigma and zruns for the original data.

    :param lcs: list of LightCurves
    :param spline: Spline, intrinsic Spline of you lcs, already optimised
    :return: 2-D array, containing the target [z_run, sigma] for each LightCurve
    """
    rls = pycs3.gen.stat.subtract(lcs, spline)
    fit_sigma = [pycs3.gen.stat.mapresistats(rls)[i]["std"] for i in range(len(rls))]
    fit_zruns = [pycs3.gen.stat.mapresistats(rls)[i]["zruns"] for i in range(len(rls))]
    fit_vector = [[fit_zruns[i], fit_sigma[i]] for i in range(len(rls))]
    return fit_vector
