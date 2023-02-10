"""
Top level functions that use the src module to tweak microlensing and source splines.
These are the function to pass them to draw.draw or draw.multidraw.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc

import pycs3.gen.lc
import pycs3.gen.lc_func
import pycs3.gen.stat
import pycs3.sim.power_spec
import pycs3.sim.src

logger = logging.getLogger(__name__)


def tweakml(lcs, spline, beta=-2.0, sigma=0.05, fmin=1 / 500.0, fmax=None, psplot=False, sampling=0.1):
    """
    I tweak the SplineML of your curves by adding small scale structure.
    I DO modify your lcs inplace.


    """
    for l in lcs:
        if l.ml is None:
            logger.warning(("Curve %s has no ML to tweak !" % (str(l))))
            continue

        elif l.ml.mltype != "spline":
            logger.warning(("I can only tweak SplineML objects, curve %s has something else !" % (str(l))))
            continue

        spline = l.ml.spline.copy()
        name = "ML(%s)" % l.object
        source = pycs3.sim.src.Source(spline, name=name, sampling=sampling)

        if psplot:
            psspline = pycs3.sim.power_spec.PowerSpectrum(source, flux=False)
            psspline.plotcolour = "black"
            psspline.calcslope(fmin=1 / 1000.0, fmax=1 / 100.0)

        source.addplaw2(beta=beta, sigma=sigma, fmin=fmin, fmax=fmax, flux=False, seed=None)
        source.name += "_twk"
        newspline = source.generate_spline()
        l.ml.replacespline(newspline)

        if psplot:
            psnewspline = pycs3.sim.power_spec.PowerSpectrum(source, flux=False)
            psnewspline.plotcolour = "red"
            psnewspline.calcslope(fmin=fmin, fmax=fmax)

            pycs3.sim.power_spec.psplot([psspline, psnewspline], nbins=50)


def tweakspl(spline, beta=-2.5, sigma=0.03, fmin=1 / 30.0, fmax=1 / 5.0, hann=False, psplot=False):
    """
    Give me a spline, I return a tweaked version with added small scale structure.
    Note that the spline I return will have a LOT of knots.

    I DO NOT modify your spline, but return a new one.
    """

    source = pycs3.sim.src.Source(spline,
                                  sampling=0.2)  # No need to pass a copy, spline will not be modified, only evaluated.

    if psplot:
        psspline = pycs3.sim.power_spec.PowerSpectrum(source, flux=False)
        psspline.plotcolour = "black"
        psspline.calcslope(fmin=1 / 1000.0, fmax=1 / 30.0)

    source.addplaw2(beta=beta, sigma=sigma, fmin=fmin, fmax=fmax, hann=hann, flux=False, seed=None)
    source.name += "_twk"
    newspline = source.generate_spline()

    if psplot:
        psnewspline = pycs3.sim.power_spec.PowerSpectrum(source, flux=False)
        psnewspline.plotcolour = "red"
        psnewspline.calcslope(fmin=fmin, fmax=fmax)

        pycs3.sim.power_spec.psplot([psspline, psnewspline], nbins=50)

    return newspline

def tweakml_PS(lcs, spline, B, f_min = 1/300.0,psplot=False, save_figure_folder = None,  verbose = False, interpolation = 'linear', A_correction = 1.0):
    """
    This function is equivalent to tweakml but I am using the power spectrum of the residuals to reinject noise with the same power spectrum
    but randomised phases. I will tweak the SplineML by adding small scale structures at the same frequencies than the data.

    I DO modify your lcs inplace.

    :param lcs: list of LightCurve
    :param spline: Spline object corresponding to the intrinsic signal of your lcs
    :param B: float, high frequency cut in units of the Nymquist frequency
    :param f_min: float, low frequency cut, in units of 1/days.
    :param psplot: boolean, Choose if you want to plot the debugging plots
    :param save_figure_folder: string. Path where to plot the figures
    :param verbose: boolean. Verbosity
    :param interpolation: string, interpolation type. Choose between 'nearest' and 'linear'
    :param A_correction: Correction factor to the amplitude of the power spectrum. To produce the same rms standard deviation in the residuals than the data I need a some small adjustment because the automatic adjustment of the amplitude is not sufficient.
    :return: Nothing, I modify the lcs.

    """
    for l in lcs:
        # We check if the attached ml really is a spline, you should change that before calling the function if this is not the case
        if l.ml == None:
            raise RuntimeError("ERROR, curve %s has no ML to tweak ! I won't tweak anything." % (str(l)))

        elif l.ml.mltype != "spline":
            raise RuntimeError("ERROR, I can only tweak SplineML objects, curve %s has something else !  I won't tweak anything." % (str(l)))

        name = "ML(%s)" % (l.object)
        ml_spline = l.ml.spline.copy()
        np.random.seed() #this is to reset the seed when using multiprocessing
        rls = pycs3.gen.stat.subtract([l], spline)[0]
        target_std = pycs3.gen.stat.resistats(rls)['std']
        target_zruns = pycs3.gen.stat.resistats(rls)['zruns']

        x = rls.jds
        y = rls.mags
        n = len(x)
        start = x[0]
        stop = x[-1]
        span = stop - start
        sampling = span / n

        sample_per_day = 5  # number of samples you want in the generated noise, the final curve is interpolated from this, choosing this too low will cut the high frequencies, and you will have too much correlated noise (too low zruns, B is going up and not converging). The high frequency can be limited by this so we adjust this value with the frequency window.
        if B >= 1 : sample_per_day = 7
        if B >= 1.5 : sample_per_day = 10
        if B >= 2. : sample_per_day = 15
        if B >= 2.5 : sample_per_day = 20
        if B >= 3.0 : sample_per_day = 30 #this is empirical... it should be a way to compute this, this is just not to cut high frequency when resampling the noise

        samples =  int(span) * sample_per_day
        if samples%2 ==1 :
            samples -= 1
        samplerate = 1 # don't touch this, add more sample if you want

        freqs_noise = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
        freqs_data = np.linspace(f_min, B* 1 / (sampling * 2.0), 10000)
        pgram = sc.lombscargle(x, y, freqs_data)

        if verbose :
            logger.info("#############################################")
            logger.info(f"Light curve {str(l.object)}")
            logger.info(f"Time Span of your lightcurve : {span} days")
            logger.info(f"Average sampling of the curve [day] : {sampling}")
            logger.info(f"Nymquist frequency [1/day]: {1 / (sampling * 2.0)}")
            logger.info(f"min max, lenght frequency of noise: {np.min(freqs_noise)}, {np.max(freqs_noise)}, {len(freqs_noise)}")
            logger.info(f"min max, lenght frequency of data: {np.min(freqs_data)}, {np.max(freqs_data)}, {len(freqs_data)}")
            logger.info(f"Number of samples generated : {samples}")

        #generate noise with not the good scaling
        band_noise = band_limited_noise_withPS(freqs_data, len(freqs_data)*pgram, samples=samples, samplerate=samplerate) #generate the noie with a PS from the data
        x_sample = np.linspace(start, stop, samples)

        noise_lcs_band = pycs3.gen.lc.LightCurve()
        noise_lcs_band.jds = x_sample
        noise_lcs_band.mags = band_noise

        #Use the previous to have the correct rescaling of the noise :
        generated_std = pycs3.gen.stat.resistats(noise_lcs_band)['std']
        Amp = target_std / generated_std
        if verbose :
            logger.info(f"required amplification : {Amp}")
            logger.info(f"Additionnal A correction : {A_correction}")
        band_noise_rescaled = band_limited_noise_withPS(freqs_data, len(freqs_data)* Amp * pgram * A_correction, samples=samples, samplerate=samplerate)
        noise_lcs_rescaled = pycs3.gen.lc.LightCurve()
        noise_lcs_rescaled.jds = x_sample
        noise_lcs_rescaled.mags = band_noise_rescaled

        #resampling of the generated noise :
        noise_lcs_resampled = pycs3.gen.lc_func.interpolate(rls, noise_lcs_rescaled, interpolate=interpolation)
        if verbose :
            logger.info(f"resampled : {pycs3.gen.stat.resistats(noise_lcs_resampled)}")
            logger.info(f"target : {pycs3.gen.stat.resistats(rls)}")


        source = pycs3.sim.src.Source(ml_spline, name=name, sampling=span/float(samples))
        if len(noise_lcs_rescaled) != len(source.imags): #weird error can happen for some curves due to round error...
            if verbose :
                logger.warning("Warning : round error somewhere, I will need to change a little bit the sampling of your source, but don't worry, I can deal with that.")
            source.sampling = float(source.jdmax - source.jdmin) / float(len(noise_lcs_rescaled))
            source.ijds = np.linspace(source.jdmin, source.jdmax, (len(noise_lcs_rescaled)))
            source.imags = source.inispline.eval(jds=source.ijds)

        source.imags += noise_lcs_rescaled.mags
        newspline = source.generate_spline()
        l.ml.replacespline(newspline) # replace the previous spline with the tweaked one...

        if psplot :
            pgram_resampled = sc.lombscargle(noise_lcs_resampled.jds, noise_lcs_resampled.mags, freqs_data)
            fig4 = plt.figure(4)
            plt.plot(freqs_data, pgram, label='original')
            plt.plot(freqs_data, pgram_resampled, label='rescaled and resampled')
            plt.xlabel('Frequencies [1/days]')
            plt.ylabel('Power')
            plt.legend(loc='best')
            if save_figure_folder == None :
                plt.show()
                pycs3.gen.stat.plotresiduals([[noise_lcs_resampled]])
                pycs3.gen.stat.plotresiduals([[rls]])
            else :
                fig4.savefig(save_figure_folder + 'PS_plot_%s.png'%l.object)
                pycs3.gen.stat.plotresiduals([[noise_lcs_resampled]], filename=save_figure_folder + 'resinoise_generated_%s.png'%l.object)
                pycs3.gen.stat.plotresiduals([[rls]], filename=save_figure_folder + 'resinoise_original_%s.png'%l.object)



"""
the 2 functions below are from `http://www.mathworks.com/matlabcentral/fileexchange/32111-fftnoise-
generate-noise-with-a-specified-power-spectrum', you find matlab code from Aslak
Grinsted, creating noise with a specified power spectrum. It can easily be
ported to python.

Copyright (c) 2011, Aslak Grinsted
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

def fftnoise(f):
    """
    Give me an array containing the power specrtum coefficients and I am generating noise py randomising
    the phases.

    :param f: 1-D array, containing the power spectrum coefficient
    :return: 1-D array, containing the power spectrum coefficient with randomised phases
    """
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
    Generate white noise for a given range of frequencies.
    I return a vector of size depending on the samples and sample rate.

    :param min_freq: minimum cut-off frequency
    :param max_freq: maximum cut-off frequency
    :param samples: integer, number of samples
    :param samplerate: float, sample rate
    :return: 1-D array containing the power spectrum
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def band_limited_noise_withPS(freqs, PS, samples=1024, samplerate=1):
    """
    Generate noise according to a given power spectrum.
    I return a vector of size depending on the samples and sample rate.

    :param freqs: 1-D array, frequencies array
    :param PS: 1-D array, power spectrum coefficients array
    :param samples: number of samples
    :param samplerate: sample rate
    :return: 1-D array containing the new power spectrum
    """
    freqs_noise = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    PS_interp = np.interp(freqs_noise, freqs, PS, left=0., right=0.)

    f = np.ones(samples) * PS_interp
    return fftnoise(f)
