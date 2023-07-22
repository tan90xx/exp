import random
import math
import warnings
from typing import Union
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi
import librosa

# Set the random seeds
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

def convert_frequency_to_mel(f: float) -> float:
    """
    Convert f hertz to mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)

def convert_mel_to_frequency(m: Union[float, np.array]) -> Union[float, np.array]:
    """
    Convert m mels to hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)

def next_power_of_2(x: int) -> int:
    """
    taken jhoyla's answer here:
    https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

class BaseTransform:

    def __init__(self):
        self.parameters = {}
        self.are_parameters_frozen = False

    def serialize_parameters(self):
        """
        Return the parameters as a JSON-serializable dict.
        Useful for when you want to store metadata on how a sound was perturbed.
        """
        return self.parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
    
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        raise NotImplementedError
    
    def apply(self, samples: np.ndarray, sample_rate: int):
        raise NotImplementedError
    
    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if samples.dtype == np.float64:
            warnings.warn(
                "Warning: input samples dtype is np.float64. Converting to np.float32"
            )
            samples = np.float32(samples)
        if not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate)
            return self.apply(samples, sample_rate)
        return samples

class BaseButterworthFilter(BaseTransform):
    """
    A `scipy.signal.butter`-based generic filter class.
    """

    # The types below must be equal to the ones accepted by
    # the `btype` argument of `scipy.signal.butter`
    ALLOWED_ONE_SIDE_FILTER_TYPES = ("lowpass", "highpass")
    ALLOWED_TWO_SIDE_FILTER_TYPES = ("bandpass", "bandstop")
    ALLOWED_FILTER_TYPES = ALLOWED_ONE_SIDE_FILTER_TYPES + ALLOWED_TWO_SIDE_FILTER_TYPES

    def __init__(self, **kwargs):
        assert "min_rolloff" in kwargs
        assert "max_rolloff" in kwargs
        assert "filter_type" in kwargs
        assert "zero_phase" in kwargs

        super().__init__()

        self.filter_type = kwargs["filter_type"]
        self.min_rolloff = kwargs["min_rolloff"]
        self.max_rolloff = kwargs["max_rolloff"]
        self.zero_phase = kwargs["zero_phase"]

        if self.zero_phase:
            assert (
                self.min_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12db/octave"
            assert (
                self.max_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12db/octave"
        else:
            assert (
                self.min_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6db/octave"
            assert (
                self.max_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6db/octave"

        assert (
            self.filter_type in BaseButterworthFilter.ALLOWED_FILTER_TYPES
        ), "Filter type must be one of: " + ", ".join(
            BaseButterworthFilter.ALLOWED_FILTER_TYPES
        )

        assert ("min_cutoff_freq" in kwargs and "max_cutoff_freq" in kwargs) or (
            "min_center_freq" in kwargs
            and "max_center_freq" in kwargs
            and "min_bandwidth_fraction" in kwargs
            and "max_bandwidth_fraction" in kwargs
        ), "Arguments for either a one-sided, or a two-sided filter must be given"

        if "min_cutoff_freq" in kwargs:
            self.initialize_one_sided_filter(
                min_cutoff_freq=kwargs["min_cutoff_freq"],
                max_cutoff_freq=kwargs["max_cutoff_freq"],
            )
        elif "min_center_freq" in kwargs:
            self.initialize_two_sided_filter(
                min_center_freq=kwargs["min_center_freq"],
                max_center_freq=kwargs["max_center_freq"],
                min_bandwidth_fraction=kwargs["min_bandwidth_fraction"],
                max_bandwidth_fraction=kwargs["max_bandwidth_fraction"],
            )

    def initialize_one_sided_filter(
        self,
        min_cutoff_freq,
        max_cutoff_freq,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        """

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

        if self.min_rolloff < 6 or self.min_rolloff % 6 != 0:
            raise ValueError(
                "min_rolloff must be 6 or greater, as well as a multiple of 6 (e.g. 6, 12, 18, 24...)"
            )
        if self.max_rolloff < 6 or self.max_rolloff % 6 != 0:
            raise ValueError(
                "max_rolloff must be 6 or greater, as well as a multiple of 6 (e.g. 6, 12, 18, 24...)"
            )
        if self.min_rolloff > self.max_rolloff:
            raise ValueError("min_rolloff must not be greater than max_rolloff")

    def initialize_two_sided_filter(
        self,
        min_center_freq,
        max_center_freq,
        min_bandwidth_fraction,
        max_bandwidth_fraction,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        """

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq
        self.min_bandwidth_fraction = min_bandwidth_fraction
        self.max_bandwidth_fraction = max_bandwidth_fraction

        if self.min_center_freq > self.max_center_freq:
            raise ValueError("min_center_freq must not be greater than max_center_freq")
        if self.min_bandwidth_fraction <= 0.0:
            raise ValueError("min_bandwidth_fraction must be a positive number")
        if self.max_bandwidth_fraction >= 2.0:
            raise ValueError(
                "max_bandwidth_fraction should be smaller than 2.0, since otherwise"
                " the low cut frequency of the band can be smaller than 0 Hz."
            )
        if self.min_bandwidth_fraction > self.max_bandwidth_fraction:
            raise ValueError(
                "min_bandwidth_fraction must not be greater than max_bandwidth_fraction"
            )

    def randomize_parameters(self, samples: np.array, sample_rate: int = None):
        if self.zero_phase:
            random_order = random.randint(
                self.min_rolloff // 12, self.max_rolloff // 12
            )
            self.parameters["rolloff"] = random_order * 12
        else:
            random_order = random.randint(self.min_rolloff // 6, self.max_rolloff // 6)
            self.parameters["rolloff"] = random_order * 6

        if self.filter_type in BaseButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_mel = np.random.uniform(
                low=convert_frequency_to_mel(self.min_cutoff_freq),
                high=convert_frequency_to_mel(self.max_cutoff_freq),
            )
            self.parameters["cutoff_freq"] = convert_mel_to_frequency(cutoff_mel)
        elif self.filter_type in BaseButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
            center_mel = np.random.uniform(
                low=convert_frequency_to_mel(self.min_center_freq),
                high=convert_frequency_to_mel(self.max_center_freq),
            )
            self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)

            bandwidth_fraction = np.random.uniform(
                low=self.min_bandwidth_fraction, high=self.max_bandwidth_fraction
            )
            self.parameters["bandwidth"] = (
                self.parameters["center_freq"] * bandwidth_fraction
            )

    def apply(self, samples: np.array, sample_rate: int = None):
        assert samples.dtype == np.float32

        if self.filter_type in BaseButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_freq = self.parameters["cutoff_freq"]
            nyquist_freq = sample_rate // 2
            if cutoff_freq > nyquist_freq:
                # Ensure that the cutoff frequency does not exceed the nyquist
                # frequency to avoid an exception from scipy
                cutoff_freq = nyquist_freq * 0.9999
            sos = butter(
                self.parameters["rolloff"] // (12 if self.zero_phase else 6),
                cutoff_freq,
                btype=self.filter_type,
                analog=False,
                fs=sample_rate,
                output="sos",
            )
        elif self.filter_type in BaseButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
            low_freq = self.parameters["center_freq"] - self.parameters["bandwidth"] / 2
            high_freq = (
                self.parameters["center_freq"] + self.parameters["bandwidth"] / 2
            )
            nyquist_freq = sample_rate // 2
            if high_freq > nyquist_freq:
                # Ensure that the upper critical frequency does not exceed the nyquist
                # frequency to avoid an exception from scipy
                high_freq = nyquist_freq * 0.9999
            sos = butter(
                self.parameters["rolloff"] // (12 if self.zero_phase else 6),
                [low_freq, high_freq],
                btype=self.filter_type,
                analog=False,
                fs=sample_rate,
                output="sos",
            )

        # The actual processing takes place here
        if len(samples.shape) == 1:
            if self.zero_phase:
                processed_samples = sosfiltfilt(sos, samples)
            else:
                processed_samples, _ = sosfilt(
                    sos, samples, zi=sosfilt_zi(sos) * samples[0]
                )
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            if self.zero_phase:
                for chn_idx in range(samples.shape[0]):
                    processed_samples[chn_idx, :] = sosfiltfilt(
                        sos, samples[chn_idx, :]
                    )
            else:
                zi = sosfilt_zi(sos)
                for chn_idx in range(samples.shape[0]):
                    processed_samples[chn_idx, :], _ = sosfilt(
                        sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                    )

        return processed_samples

class LowPassFilter(BaseButterworthFilter):
    """
    Apply low-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).
    """

    def __init__(
        self,
        min_cutoff_freq: float = 150.0,
        max_cutoff_freq: float = 7500.0,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `true` it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs. 3 dB). Additionally,
            it is 2 times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `true`.
        """ 
        super().__init__(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            filter_type="lowpass",
        )
    
class HighPassFilter(BaseButterworthFilter):
    """
    Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).
    """

    def __init__(
        self,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `true` it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs. 3 dB). Additionally,
            it is 2 times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `true`.
        :param p: The probability of applying this transform
        """
        super().__init__(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            filter_type="highpass",
        )
    
class BandPassFilter(BaseButterworthFilter):
    """
    Apply band-pass filtering to the input audio. Filter steepness (6/12/18... dB / octave)
    is parametrized. Can also be set for zero-phase filtering (will result in a 6 dB drop at
    cutoffs).
    """

    def __init__(
        self,
        min_center_freq: float = 200.0,
        max_center_freq: float = 4000.0,
        min_bandwidth_fraction: float = 0.5,
        max_bandwidth_fraction: float = 1.99,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth relative to center frequency
        :param max_bandwidth_fraction: Maximum bandwidth relative to center frequency
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `True` it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs 3 dB). Additionally,
            it is 2 times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment an
            audio file with lots of transients, like a drum track), set
            this to `True`.
        """
        super().__init__(
            min_center_freq=min_center_freq,
            max_center_freq=max_center_freq,
            min_bandwidth_fraction=min_bandwidth_fraction,
            max_bandwidth_fraction=max_bandwidth_fraction,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            filter_type="bandpass",
        )

class BandStopFilter(BaseButterworthFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter or
    band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
    Center frequency gets picked in mel space, so it is
    more aligned with human hearing, which is not linear. Filter steepness
    (6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
    (will result in a 6 dB drop at cutoffs).
    """

    def __init__(
        self,
        min_center_freq: float = 200.0,
        max_center_freq: float = 4000.0,
        min_bandwidth_fraction: float = 0.5,
        max_bandwidth_fraction: float = 1.99,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0 and 2)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0 and 2)
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `true` it will not affect the phase of the
            input signal but will sound 3db lower at the cutoff frequency
            compared to the non-zero phase case (6db vs 3db). Additionally,
            it is 2 times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `true`.
        :param p: The probability of applying this transform
        """
        super().__init__(
            min_center_freq=min_center_freq,
            max_center_freq=max_center_freq,
            min_bandwidth_fraction=min_bandwidth_fraction,
            max_bandwidth_fraction=max_bandwidth_fraction,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            filter_type="bandstop",
        )

class PeakingFilter(BaseTransform):
    """
    Peaking filter transform. Applies a peaking filter at a specific center frequency in hertz
    of a specific gain in dB (note: can be positive or negative!), and a quality factor
    parameter. Filter coefficients are taken from the W3 Audio EQ Cookbook:
    https://www.w3.org/TR/audio-eq-cookbook/
    """

    def __init__(
        self,
        min_center_freq: float = 50.0,
        max_center_freq: float = 7500.0,
        min_gain_db: float = -24.0,
        max_gain_db: float = 24.0,
        min_q: float = 0.5,
        max_q: float = 5.0,
    ):
        """
        :param min_center_freq: The minimum center frequency of the peaking filter
        :param max_center_freq: The maximum center frequency of the peaking filter
        :param min_gain_db: The minimum gain at center frequency in dB
        :param max_gain_db: The maximum gain at center frequency in dB
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert min_q > 0, "`min_q` should be greater than 0"
        assert max_q > 0, "`max_q` should be greater than 0"

        super().__init__()

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = 1 + alpha * gain
        b1 = -2 * np.cos(normalized_frequency)
        b2 = 1 - alpha * gain

        a0 = 1 + alpha / gain
        a1 = -2 * np.cos(normalized_frequency)
        a2 = 1 - alpha / gain

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples: np.ndarray, sample_rate: int):
        assert samples.dtype == np.float32

        sos = self._get_biquad_coefficients_from_input_parameters(
            self.parameters["center_freq"],
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples

class NotchFilter(BaseTransform):
    """
    Peaking filter transform. Applies a peaking filter at a specific center frequency in hertz
    of a specific gain in dB (note: can be positive or negative!), and a quality factor
    parameter. Filter coefficients are taken from the W3 Audio EQ Cookbook:
    https://www.w3.org/TR/audio-eq-cookbook/
    """

    def __init__(
        self,
        min_center_freq: float = 50.0,
        max_center_freq: float = 7500.0,
        min_q: float = 0.5,
        max_q: float = 5.0,
    ):
        """
        :param min_center_freq: The minimum center frequency of the peaking filter
        :param max_center_freq: The maximum center frequency of the peaking filter
        :param min_gain_db: The minimum gain at center frequency in dB
        :param max_gain_db: The maximum gain at center frequency in dB
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"

        assert min_q > 0, "`min_q` should be greater than 0"
        assert max_q > 0, "`max_q` should be greater than 0"

        super().__init__()

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = 1
        b1 = -2 * np.cos(normalized_frequency)
        b2 = 1

        a0 = 1 + alpha
        a1 = -2 * np.cos(normalized_frequency)
        a2 = 1 - alpha

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples: np.ndarray, sample_rate: int):
        assert samples.dtype == np.float32

        sos = self._get_biquad_coefficients_from_input_parameters(
            self.parameters["center_freq"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples

class LowShelfFilter(BaseTransform):
    """
    A low shelf filter is a filter that either boosts (increases amplitude) or cuts
    (decreases amplitude) frequencies below a certain center frequency. This transform
    applies a low-shelf filter at a specific center frequency in hertz.
    The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
    Filter coefficients are taken from the W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/
    """

    def __init__(
        self,
        min_center_freq: float = 50.0,
        max_center_freq: float = 24000.0,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        min_q: float = 0.1,
        max_q: float = 0.999,
    ):

        """
        :param min_center_freq: The minimum center frequency of the shelving filter
        :param max_center_freq: The maximum center frequency of the shelving filter
        :param min_gain_db: The minimum gain at DC (0 hz)
        :param max_gain_db: The maximum gain at DC (0 hz)
        :param min_q: The minimum quality factor q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert 0 < min_q <= 1, "`min_q` should be greater than 0 and less or equal to 1"
        assert 0 < max_q <= 1, "`max_q` should be greater than 0 and less or equal to 1"

        super().__init__()

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        b1 = 2 * gain * ((gain - 1) - (gain + 1) * np.cos(normalized_frequency))

        b2 = gain * (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        a0 = (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        a1 = -2 * ((gain - 1) + (gain + 1) * np.cos(normalized_frequency))

        a2 = (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples: np.ndarray, sample_rate: int):
        nyquist_freq = sample_rate // 2
        center_freq = self.parameters["center_freq"]
        if center_freq > nyquist_freq:
            # Ensure that the center frequency is below the nyquist
            # frequency to avoid filter instability
            center_freq = nyquist_freq * 0.9999

        sos = self._get_biquad_coefficients_from_input_parameters(
            center_freq,
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples

class HighShelfFilter(BaseTransform):
    """
    A high shelf filter is a filter that either boosts (increases amplitude) or cuts
    (decreases amplitude) frequencies above a certain center frequency. This transform
    applies a high-shelf filter at a specific center frequency in hertz.
    The gain at nyquist frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
    Filter coefficients are taken from the W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/
    """

    def __init__(
        self,
        min_center_freq: float = 300.0,
        max_center_freq: float = 24000.0,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        min_q: float = 0.1,
        max_q: float = 0.999,
    ):
        """
        :param min_center_freq: The minimum center frequency of the shelving filter
        :param max_center_freq: The maximum center frequency of the shelving filter
        :param min_gain_db: The minimum gain at the nyquist frequency
        :param max_gain_db: The maximum gain at the nyquist frequency
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param p: The probability of applying this transform
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert 0 < min_q <= 1, "`min_q` should be greater than 0 and less or equal to 1"
        assert 0 < max_q <= 1, "`max_q` should be greater than 0 and less or equal to 1"

        super().__init__()

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        b1 = -2 * gain * ((gain - 1) + (gain + 1) * np.cos(normalized_frequency))

        b2 = gain * (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        a0 = (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        a1 = 2 * ((gain - 1) - (gain + 1) * np.cos(normalized_frequency))

        a2 = (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        #super().randomize_parameters(samples, sample_rate)

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples: np.ndarray, sample_rate: int):
        nyquist_freq = sample_rate // 2
        center_freq = self.parameters["center_freq"]
        if center_freq > nyquist_freq:
            # Ensure that the center frequency is below the nyquist
            # frequency to avoid filter instability
            center_freq = nyquist_freq * 0.9999

        sos = self._get_biquad_coefficients_from_input_parameters(
            center_freq,
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples

class AirAbsorption(BaseTransform):
    """
    Apply a Lowpass-like filterbank with variable octave attenuation that simulates attenuation of
    higher frequencies due to air absorption. This transform is parametrized by temperature,
    humidity, and the distance between audio source and microphone.

    This is not a scientifically accurate transform but basically applies a uniform
    filterbank with attenuations given by:

    att = exp(- distance * absorption_coefficient)

    where distance is the microphone-source assumed distance in meters and `absorption_coefficient`
    is adapted from a lookup table by pyroomacoustics [1]. It can also be seen as a lowpass filter
    with variable octave attenuation.

    Note: This only "simulates" the dampening of high frequencies, and does not
    attenuate according to the distance law. Gain augmentation needs to be done separately.

    [1] https://github.com/LCAV/pyroomacoustics
    """

    # Table of air absorption coefficients adapted from `pyroomacoustics`.
    # The keys are of the form:
    #   "<degrees>C_<minimum_humidity>-<maximum_humidity>%"
    #
    # And the values are attenuation coefficients `coef` that attenuate the corresponding band
    # in "center_freq" by exp(-coef * <microphone-source distance>).
    # The original table does not have the last two columns which have been extrapolated from the
    # pyroomacoustics table using `curve_fit`
    air_absorption_table = {
        "10C_30-50%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0]
        ],
        "10C_50-70%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2]
        ],
        "10C_70-90%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9]
        ],
        "20C_30-50%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9]
        ],
        "20C_50-70%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7]
        ],
        "20C_70-90%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8]
        ],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
    }

    def __init__(
        self,
        min_temperature: float = 10.0,
        max_temperature: float = 20.0,
        min_humidity: float = 30.0,
        max_humidity: float = 90.0,
        min_distance: float = 10.0,
        max_distance: float = 100.0,
    ):
        """
        :param min_temperature: Minimum temperature in Celsius (can take a value of either 10.0 or 20.0)
        :param max_temperature: Maximum temperature in Celsius (can take a value of either 10.0 or 20.0)
        :param min_humidity: Minimum humidity in percent (between 30 and 90)
        :param max_humidity: Maximum humidity in percent (between 30 and 90)
        :param min_distance: Minimum microphone-source distance in meters.
        :param max_distance: Maximum microphone-source distance in meters.
        :param p: The probability of applying this transform
        """
        assert float(min_temperature) in [
            10.0,
            20.0,
        ], "Sorry, the only supported temperatures are either 10 or 20 degrees Celsius"
        assert float(max_temperature) in [
            10.0,
            20.0,
        ], "Sorry, the only supported temperatures are either 10 or 20 degrees Celsius"
        assert min_temperature <= max_temperature
        assert 30 <= min_humidity <= max_humidity <= 90
        assert min_distance > 0.0
        assert max_distance > 0.0
        assert min_distance <= max_distance

        super().__init__()

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self.min_humidity = min_humidity
        self.max_humidity = max_humidity

        self.min_distance = min_distance
        self.max_distance = max_distance

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.parameters["temperature"] = 10 * np.random.randint(
            int(self.min_temperature) // 10, int(self.max_temperature) // 10 + 1
        )
        self.parameters["humidity"] = np.random.randint(
            self.min_humidity, self.max_humidity + 1
        )
        self.parameters["distance"] = np.random.uniform(
            self.min_distance, self.max_distance
        )

    def apply(self, samples: np.ndarray, sample_rate: int):
        assert samples.dtype == np.float32

        humidity = self.parameters["humidity"]
        distance = self.parameters["distance"]

        # Choose correct absorption coefficients
        key = str(int(self.parameters["temperature"])) + "C"
        bounds = [30, 50, 70, 90]
        for n in range(1, len(bounds)):
            if bounds[n - 1] <= humidity or humidity <= bounds[n]:
                key += f"_{bounds[n-1]}-{bounds[n]}%"
                break

        # Convert to attenuations
        attenuation_values = np.exp(
            -distance * np.array(self.air_absorption_table[key])
        )

        # Calculate n_fft so that the lowest band can be stored in a single
        # fft bin.
        first_band_bw = self.air_absorption_table["center_freqs"][0] / (2**0.5)
        n_fft = next_power_of_2(int(sample_rate / 2 / first_band_bw))

        # Frequencies to calculate the attenuations caused by air absorption
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        # Interpolate to the desired frequencies (we have to do this in dB)
        db_target_attenuations = np.interp(
            frequencies,
            self.air_absorption_table["center_freqs"],
            20 * np.log10(attenuation_values),
        )

        linear_target_attenuations = 10 ** (db_target_attenuations / 20)

        # Apply using STFT
        if len(samples.shape) == 1:
            stft = librosa.stft(samples, n_fft=n_fft)

            # Compute mask
            mask = np.tile(linear_target_attenuations, (stft.shape[1], 1)).T

            # Compute target degraded audio
            result = librosa.istft(stft * mask, length=len(samples), dtype=np.float32)

        else:
            result = np.zeros_like(samples, dtype=np.float32)

            for chn_idx, channel in enumerate(samples):
                stft = librosa.stft(channel, n_fft=n_fft)

                # Compute mask
                mask = np.tile(linear_target_attenuations, (stft.shape[1], 1)).T

                # Compute target degraded audio
                result[chn_idx, :] = librosa.istft(stft * mask, length=result.shape[1])

        return result
    
class SpecFrequencyMask(BaseTransform):
    """
    Mask a set of frequencies in a spectrogram, à la Google AI SpecAugment. This type of data
    augmentation has proved to make speech recognition models more robust.

    The masked frequencies can be replaced with either the mean of the original values or a
    given constant (e.g. zero).
    """

    def __init__(
        self,
        min_mask_fraction: float = 0.5,
        max_mask_fraction: float = 0.99,
        fill_mode: str = "constant",
        fill_constant: float = 0.0,
    ):
        super().__init__()
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.fill_constant = fill_constant

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        num_frequency_bins = sample_rate
        min_frequencies_to_mask = int(
            round(self.min_mask_fraction * num_frequency_bins)
        )
        max_frequencies_to_mask = int(
            round(self.max_mask_fraction * num_frequency_bins)
        )
        num_frequencies_to_mask = random.randint(
            min_frequencies_to_mask, max_frequencies_to_mask
        )
        self.parameters["start_frequency_index"] = random.randint(
            0, num_frequency_bins - num_frequencies_to_mask
        )
        self.parameters["end_frequency_index"] = (
            self.parameters["start_frequency_index"] + num_frequencies_to_mask
        )

    def apply(self, samples: np.ndarray, sample_rate: int):
        n_fft = sample_rate
        magnitude_spectrogram = librosa.stft(samples, n_fft=n_fft)
        if self.fill_mode == "mean":
            fill_value = np.mean(
                magnitude_spectrogram[
                self.parameters["start_frequency_index"] : self.parameters[
                    "end_frequency_index"
                ]
                ]
            )
        else:
            # self.fill_mode == "constant"
            fill_value = self.fill_constant
        magnitude_spectrogram = magnitude_spectrogram.copy()
        magnitude_spectrogram[
        self.parameters["start_frequency_index"] : self.parameters[
            "end_frequency_index"
        ]
        ] = fill_value
        result = librosa.istft(magnitude_spectrogram, length=len(samples), dtype=np.float32)
        return result