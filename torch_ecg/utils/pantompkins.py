"""
this file is borrowed and modified from wfdb 2.2.1, which is removed in wfdb 3.*.*

source: https://pypi.org/project/wfdb/2.2.1/#files

"""
import numpy as np
import scipy.signal as scisig

# import pdb
# import matplotlib.pyplot as plt


__all__ = [
    "PanTompkins",
    "pantompkins",
]


class PanTompkins(object):
    """
    Class for implementing the Pan-Tompkins
    qrs detection algorithm.
    """

    def __init__(self, sig=None, fs=None, streamsig=None, verbose=0):
        self.sig = sig
        self.fs = fs
        self.verbose = verbose

        self.streamsig = streamsig

        if sig is not None:
            self.siglen = len(sig)

        # Feature to add
        # "For irregular heart rates, the first threshold
        # of each set is reduced by half so as to increase
        # the detection sensitivity and to avoid missing beats"
        # self.irregular_hr = irregular_hr

    def ispeak(self, sig, siglen, ind, windonw):
        return sig[ind] == (sig[ind - windonw : ind + windonw]).max()

    def detect_qrs_static(self):
        """
        Detect all the qrs locations in the static signal
        """

        # Resample the signal to 200Hz if necessary
        self.resample()

        # Bandpass filter the signal
        self.bandpass(plotsteps=False)

        # Calculate the moving wave integration signal
        self.mwi(plotsteps=False)

        # Align the filtered and integrated signal with the original
        self.alignsignals()

        # Let's do some investigating!
        # Compare sig, sig_F, and sig_I

        # 1. Compare sig_F and sig_I peaks

        fpeaks = findpeaks_radius(self.sig_F, 20)

        ipeaks = findpeaks_radius(self.sig_I, 20)

        fpeaks = fpeaks[np.where(self.sig_F[fpeaks] > 4)[0]]
        ipeaks = ipeaks[np.where(self.sig_I[ipeaks] > 4)[0]]

        # allpeaks = np.union1d(fpeaks, ipeaks)

        if self.verbose >= 1:
            print("fpeaks:", fpeaks)
            print("ipeaks:", ipeaks)

        if self.verbose >= 2:
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(self.sig_F, "b")
            plt.plot(fpeaks, self.sig_F[fpeaks], "b*")

            plt.plot(self.sig_I, "r")
            plt.plot(ipeaks, self.sig_I[ipeaks], "r*")

            # plt.plot(allpeaks, self.sig_F[allpeaks], "g*")

            plt.show()

        # Initialize learning parameters via the two learning phases
        self.learnparams()

        # Loop through every index and detect qrs locations.
        # Start from 200ms after the first qrs detected in the
        # learning phase
        for i in range(self.qrs_inds[0] + 41, self.siglen):

            # Number of indices from the previous r peak to this index
            last_r_distance = i - self.qrs_inds[-1]

            # It has been very long since the last r peak
            # was found. Search back for common 2 signal
            # peaks and test using lower threshold
            if last_r_distance > self.rr_missed_limit:

                self.backsearch()
                # Continue with this index whether or not
                # a previous common peak was marked as qrs
                last_r_distance = i - self.qrs_inds[-1]

            # Determine whether the current index is a peak
            # for each signal
            is_peak_F = self.ispeak(self.sig_F, self.siglen, i, 20)
            is_peak_I = self.ispeak(self.sig_I, self.siglen, i, 20)

            # Keep track of common peaks that have not been classified as
            # signal peaks for future backsearch
            if is_peak_F and is_peak_I:
                self.recent_commonpeaks.append(i)

            # Whether the current index is a signal peak or noise peak
            is_sigpeak_F = False
            is_sigpeak_I = False

            # If peaks are detected, classify them as signal or noise
            # for their respective channels

            if is_peak_F:
                # Satisfied signal peak criteria for the channel
                if self.sig_F[i] > self.thresh_F:
                    is_sigpeak_F = True
                # Did not satisfy signal peak criteria.
                # Classify as noise peak
                else:
                    self.update_peak_params("nF", i)

            if is_peak_I:
                # Satisfied signal peak criteria for the channel
                if self.sig_I[i] > self.thresh_I:
                    is_sigpeak_I = True
                # Did not satisfy signal peak criteria.
                # Classify as noise peak
                else:
                    self.update_peak_params("nI", i)

            # Check for double signal peak coincidence and at least >200ms (40 samples samples)
            # since the previous r peak
            is_sigpeak = is_sigpeak_F and is_sigpeak_I and (last_r_distance > 40)

            # The peak crosses thresholds for each channel and >200ms from previous peak
            if is_sigpeak:
                # If the rr interval < 360ms (72 samples), the peak is checked
                # to determine whether it is a T-Wave. This is the final test
                # to run before the peak can be marked as a qrs complex.
                if last_r_distance < 72:
                    is_twave = self.istwave(i)
                    # Classified as likely a t-wave, not a qrs.
                    if is_twave:
                        self.update_peak_params("nF", i)
                        self.update_peak_params("nI", i)
                        continue

                # Finally can classify as a signal peak
                # Update running parameters
                self.update_peak_params("ss", i)

                continue

            # No double agreement of signal peak.
            # Any individual peak that passed its threshold criterial
            # will still be classified as noise peak.
            elif is_sigpeak_F:
                self.update_peak_params("nF", i)
            elif is_sigpeak_I:
                self.update_peak_params("nI", i)

        # Convert the peak indices back to the original fs if necessary
        self.reverseresampleqrs()

        return

    def resample(self):
        if self.fs != 200:
            self.sig = scisig.resample(self.sig, int(self.siglen * 200 / self.fs))
        return

    # Bandpass filter the signal from 5-15Hz
    def bandpass(self, plotsteps=False):
        # 15Hz Low Pass Filter
        a_low = [1, -2, 1]
        b_low = np.concatenate(([1], np.zeros(4), [-2], np.zeros(5), [1]))
        sig_low = scisig.lfilter(b_low, a_low, self.sig)

        # 5Hz High Pass Filter - passband gain = 32, delay = 16 samples
        a_high = [1, -1]
        b_high = np.concatenate(
            ([-1 / 32], np.zeros(15), [1, -1], np.zeros(14), [1 / 32])
        )
        self.sig_F = scisig.lfilter(b_high, a_high, sig_low)

        if plotsteps:
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            plt.plot(sig_low)
            plt.plot(self.sig_F)
            plt.legend(["After LP", "After LP+HP"])
            plt.show()
        return

    # Compute the moving wave integration waveform from the filtered signal
    def mwi(self, plotsteps=False):
        # Compute 5 point derivative
        a_deriv = [1]
        b_deriv = [1 / 4, 1 / 8, 0, -1 / 8, -1 / 4]
        sig_F_deriv = scisig.lfilter(b_deriv, a_deriv, self.sig_F)

        # Square the derivative
        sig_F_deriv = np.square(sig_F_deriv)

        # Perform moving window integration - 150ms (ie. 30 samples wide for 200Hz)
        a_mwi = [1]
        b_mwi = 30 * [1 / 30]

        self.sig_I = scisig.lfilter(b_mwi, a_mwi, sig_F_deriv)

        if plotsteps:
            if "plt" not in dir():
                import matplotlib.pyplot as plt
            # plt.plot(sig_deriv)
            plt.plot(self.sig_I)
            plt.legend(["deriv", "mwi"])
            plt.show()
        return

    # Align the filtered and integrated signal with the original
    def alignsignals(self):
        self.sig_F = self.sig_F

        self.sig_I = self.sig_I

        return

    def learnparams(self):
        """
        Initialize detection parameters using the start of the waveforms
        during the two learning phases described.

        "Learning phase 1 requires about 2s to initialize
        detection thresholds based upon signal and noise peaks
        detected during the learning process.

        Learning phase two requires two heartbeats to initialize
        RR-interval average and RR-interval limit values.

        The subsequent detection phase does the recognition process
        and produces a pulse for each QRS complex"

        This code is not detailed in the Pan-Tompkins
        paper. The PT algorithm requires a threshold to
        categorize peaks as signal or noise, but the
        threshold is calculated from noise and signal
        peaks. There is a circular dependency when
        none of the fields are initialized. Therefore this
        learning phase will detect initial signal peaks using a
        different method, and estimate the threshold using
        those peaks.

        This function works as follows:
        - Try to find at least 2 signal peaks (qrs complexes) in the
          first 2 seconds of both signals using simple low order
          moments. Signal peaks are only defined when the same index is
          determined to be a peak in both signals. If fewer than 2 signal
          peaks are detected, shift to the next 2 window and try again.
        - Using the classified estimated peaks, threshold is estimated as
          based on the steady state estimate equation: thres = 0.75*noisepeak + 0.25*sigpeak
          using the mean of the noisepeaks and signalpeaks instead of the
          running value.
        - Using the estimated peak locations, the rr parameters are set.

        """

        # The sample radius when looking for local maxima
        radius = 20
        # The signal start duration to use for learning
        learntime = 2
        # The window number to inspect in the signal
        windownum = 0

        while (windownum + 1) * learntime * 200 < self.siglen:
            wavelearn_F = self.sig_F[
                windownum * learntime * 200 : (windownum + 1) * learntime * 200
            ]
            wavelearn_I = self.sig_I[
                windownum * learntime * 200 : (windownum + 1) * learntime * 200
            ]

            # Find peaks in the signal sections
            peakinds_F = findpeaks_radius(wavelearn_F, radius)
            peakinds_I = findpeaks_radius(wavelearn_I, radius)
            peaks_F = wavelearn_F[peakinds_F]
            peaks_I = wavelearn_I[peakinds_I]

            # Classify signal and noise peaks.
            # This is the main tricky part

            # Align peaks to minimum value and set to unit variance
            peaks_F = (peaks_F - min(peaks_F)) / np.std(peaks_F)
            peaks_I = (peaks_I - min(peaks_I)) / np.std(peaks_I)
            sigpeakinds_F = np.where(peaks_F >= 1.4)
            sigpeakinds_I = np.where(peaks_I >= 1.4)

            # Final signal peak when both signals agree
            sigpeakinds = np.intersect1d(sigpeakinds_F, sigpeakinds_I)
            # Noise peaks are the remainders
            noisepeakinds_F = np.setdiff1d(peakinds_F, sigpeakinds)
            noisepeakinds_I = np.setdiff1d(peakinds_I, sigpeakinds)

            # Found at least 2 peaks. Also peak 1 and 2 must be >200ms apart
            if len(sigpeakinds) > 1 and sigpeakinds[1] - sigpeakinds[0] > 40:
                print("should be out")
                break

            # Didn't find 2 satisfactory peaks. Check the next window.
            windownum = windownum + 1

        # Found at least 2 satisfactory peaks. Use them to set parameters.

        # Set running peak estimates to first values
        self.sigpeak_F = wavelearn_F[sigpeakinds[0]]
        self.sigpeak_I = wavelearn_I[sigpeakinds[0]]
        self.noisepeak_F = wavelearn_F[noisepeakinds_F[0]]
        self.noisepeak_I = wavelearn_I[noisepeakinds_I[0]]

        # Use all signal and noise peaks in learning window to estimate threshold
        # Based on steady state equation: thres = 0.75*noisepeak + 0.25*sigpeak
        self.thresh_F = 0.75 * np.mean(wavelearn_F[noisepeakinds_F]) + 0.25 * np.mean(
            wavelearn_F[sigpeakinds_F]
        )
        self.thresh_I = 0.75 * np.mean(wavelearn_I[noisepeakinds_I]) + 0.25 * np.mean(
            wavelearn_I[sigpeakinds_I]
        )
        # Alternatively, could skip all of that and do something very simple like thresh_F =  max(filtsig[:400])/3

        # Set the r-r history using the first r-r interval
        # The most recent 8 rr intervals
        self.rr_history_unbound = [
            wavelearn_F[sigpeakinds[1]] - wavelearn_F[sigpeakinds[0]]
        ] * 8
        # The most recent 8 rr intervals that fall within the acceptable low and high rr interval limits
        self.rr_history_bound = [
            wavelearn_I[sigpeakinds[1]] - wavelearn_I[sigpeakinds[0]]
        ] * 8

        self.rr_average_unbound = np.mean(self.rr_history_unbound)
        self.rr_average_bound = np.mean(self.rr_history_bound)

        # what is rr_average_unbound ever used for?
        self.rr_low_limit = 0.92 * self.rr_average_bound
        self.rr_high_limit = 1.16 * self.rr_average_bound
        self.rr_missed_limit = 1.66 * self.rr_average_bound

        # The qrs indices detected.
        # Initialize with the first signal peak
        # detected during this learning phase
        self.qrs_inds = [sigpeakinds[0]]

        return

    # Update parameters when a peak is found
    def update_peak_params(self, peaktype, i):

        # Noise peak for filtered signal
        if peaktype == "nF":
            self.noisepeak_F = 0.875 * self.noisepeak_I + 0.125 * self.sig_I[i]
        # Noise peak for integral signal
        elif peaktype == "nI":
            self.noisepeak_I = 0.875 * self.noisepeak_I + 0.125 * self.sig_I[i]
        # Signal peak
        else:
            new_rr = i - self.qrs_inds[-1]

            # The most recent 8 rr intervals
            self.rr_history_unbound = self.rr_history_unbound[:-1].append(new_rr)
            self.rr_average_unbound = self.rr_average_unbound[:-1]

            # The most recent 8 rr intervals that fall within the acceptable low
            # and high rr interval limits
            if new_rr > self.r_low_limit and new_rr < self.r_high_limit:
                self.rr_history_bound = self.rr_history_bound[:-1].append(new_rr)
                self.rr_average_bound = np.mean(self.rr_history_bound)

                self.rr_low_limit = 0.92 * self.rr_average_bound
                self.rr_high_limit = 1.16 * self.rr_average_bound
                self.rr_missed_limit = 1.66 * self.rr_average_bound

            # Clear the common peaks since last r peak variable
            self.recent_commonpeaks = []

            self.qrs_inds.append(i)

            # Signal peak, regular threshold criteria
            if peaktype == "sr":
                self.sigpeak_I = 0.875 * self.sigpeak_I + 0.125 * self.sig_I[i]
                self.sigpeak_F = 0.875 * self.sigpeak_F + 0.125 * self.sig_F[i]
            else:
                # Signal peak, searchback criteria
                self.sigpeak_I = 0.75 * self.sigpeak_I + 0.25 * self.sig_I[i]
                self.sigpeak_F = 0.75 * self.sigpeak_F + 0.25 * self.sig_F[i]

        return

    def backsearch(self):
        """
        Search back for common 2 signal
        peaks and test for qrs using lower thresholds

        "If the program does not find a QRS complex in
        the time interval corresponding to 166 percent
        of the current average RR interval, the maximal
        peak deteted in that time interval that lies
        between these two thresholds is considered to be
        a possilbe QRS complex, and the lower of the two
        thresholds is applied"

        Interpreting the above "the maximal peak":
        - A common peak in both sig_F and sig_I.
        - The largest sig_F peak.
        """

        # No common peaks since the last r
        if not self.recent_commonpeaks:
            return

        recentpeaks_F = self.sig_F[self.recent_commonpeaks]

        # Overall signal index to inspect
        maxpeak_ind = self.recent_commonpeaks[np.argmax(recentpeaks_F)]

        # Test these peak values
        sigpeak_F = self.sig_F[maxpeak_ind]
        sigpeak_I = self.sig_I[maxpeak_ind]

        # Thresholds passed for both signals. Found qrs.
        if (sigpeak_F > self.thresh_F / 2) and (sigpeak_I > self.thresh_I / 2):
            self.update_peak_params("ss", maxpeak_ind)

        return

    # QRS duration between 0.06s and 0.12s
    # Check left half - 0.06s = 12 samples
    qrscheckwidth = 12
    # ST segment between 0.08 and 0.12s.
    # T-wave duration between 0.1 and 0.25s.
    # We are only analyzing left half for gradient
    # Overall, check 0.12s to the left of the peak.
    # tcheckwidth = 24

    def istwave(self, i):
        """
        Determine whether the coinciding peak index happens
        to be a t-wave instead of a qrs complex.

        "If the maximal slope that occurs during this waveform
        is less than half that of the QRS waveform that preceded
        it, it is identified to be a T wave"

        Compare slopes in filtered signal only
        """

        # Parameter: Checking width of a qrs complex
        # Parameter: Checking width of a t-wave

        a_deriv = [1]
        b_deriv = [1 / 4, 1 / 8, 0, -1 / 8, -1 / 4]

        lastqrsind = self.qrs_inds[:-1]

        qrs_sig_F_deriv = scisig.lfilter(
            b_deriv, a_deriv, self.sig_F[lastqrsind - self.qrscheckwidth : lastqrsind]
        )
        checksection_sig_F_deriv = scisig.lfilter(
            b_deriv, a_deriv, self.sig_F[i - self.qrscheckwidth : i]
        )

        # Classified as a t-wave
        if max(checksection_sig_F_deriv) < 0.5 * max(qrs_sig_F_deriv):
            return True
        else:
            return False

    def reverseresampleqrs(self):
        # Refactor the qrs indices to match the fs of the original signal

        self.qrs_inds = np.array(self.qrs_inds)

        if self.fs != 200:
            self.qrs_inds = self.qrs_inds * self.fs / 200

        self.qrs_inds = self.qrs_inds.astype("int64")


def pantompkins(sig, fs):
    """
    Pan Tompkins ECG peak detector
    """
    detector = PanTompkins(sig=sig, fs=fs)
    detector.detect_qrs_static()

    return detector.qrs_inds


# Determine whether the signal contains a peak at index ind.
# Check if it is the max value amoung samples ind-radius to ind+radius
def ispeak_radius(sig, siglen, ind, radius):
    if sig[ind] == max(sig[max(0, ind - radius) : min(siglen, ind + radius)]):
        return True
    else:
        return False


# Find all peaks in a signal. Simple algorithm which marks a
# peak if the <radius> samples on its left and right are
# all not bigger than it.
# Faster than calling ispeak_radius for every index.
def findpeaks_radius(sig, radius):

    siglen = len(sig)
    peaklocs = []

    # Pad samples at start and end
    sig = np.concatenate((np.ones(radius) * sig[0], sig, np.ones(radius) * sig[-1]))

    i = radius
    while i < siglen + radius:
        if sig[i] == max(sig[i - radius : i + radius]):
            peaklocs.append(i)
            i = i + radius
        else:
            i = i + 1

    peaklocs = np.array(peaklocs) - radius
    return peaklocs
