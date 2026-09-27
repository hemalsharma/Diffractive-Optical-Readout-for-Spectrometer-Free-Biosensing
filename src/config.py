"""
config.py -- Stage 1 simulation parameters, shared by main.py and validation.py.

Geometry choice. A 256 x 256 DOE with 10 um pixels is a 2.56 mm aperture.
The grating period (160 um = 16 pixels) and distance (50 mm) are chosen so
that the +1 order (x = z tan(asin(lambda/d)) ~ 0.48 mm at 1550 nm) is well
separated from the axis (> 2 beam radii) yet stays inside the periodic
FFT window over the whole 1450-1650 nm validation sweep.
"""

from dataclasses import dataclass

from sensor_model import PhysicalUnits, ResonanceParams, SensorModel


@dataclass(frozen=True)
class Config:
    # Sensor (physical)
    lambda0_nm: float = 1550.0       # zero-analyte resonance
    fwhm_nm: float = 5.0             # resonance FWHM (HWHM gamma = 2.5 nm)
    depth: float = 0.8               # resonance depth A
    shift_per_c_nm: float = 2.5      # alpha: one unit of c shifts by one HWHM
    source_fwhm_nm: float = 20.0     # Gaussian (SLED-like) source
    c_max: float = 2.0               # largest concentration simulated
    window_gammas: float = 12.0      # spectral window beyond the resonance
    n_lambda: int = 150              # wavelength samples

    # Optics
    n_doe: int = 256                 # DOE pixels per side
    pitch: float = 10e-6             # DOE pixel pitch [m]
    upsample: int = 2                # simulation samples per DOE pixel
    pad: int = 1                     # window / DOE size
    w0: float = 200e-6               # Gaussian 1/e^2 intensity radius [m]
    z: float = 50e-3                 # DOE -> detector distance [m]
    period: float = 160e-6           # validation grating period [m]

    # Two-detector split around the +1 order (preview of Stage 2 readout)
    det_width: float = 300e-6
    det_height: float = 600e-6

    @property
    def units(self):
        return PhysicalUnits(lambda0_nm=self.lambda0_nm)

    @property
    def resonance(self):
        """Normalized resonance parameters (lambda0 = 1)."""
        g = 0.5 * self.fwhm_nm / self.lambda0_nm
        return ResonanceParams(lambda0=1.0, gamma=g, depth=self.depth,
                               alpha=self.shift_per_c_nm / self.lambda0_nm)

    def sensor(self, n_lambda=None):
        p = self.resonance
        return SensorModel.build(
            p, c_range=(0.0, self.c_max), n=n_lambda or self.n_lambda,
            half_width_gammas=self.window_gammas, source="gaussian",
            source_fwhm_gammas=self.source_fwhm_nm / (0.5 * self.fwhm_nm))

    def lam_m(self, sensor):
        """Sensor wavelength grid in metres."""
        return sensor.lam * self.lambda0_nm * 1e-9

    def dlam_nm(self, sensor):
        return float(sensor.dlam) * self.lambda0_nm


CFG = Config()


@dataclass(frozen=True)
class Embodiment2Config:
    """Embodiment 2 (pixel-wise spectral coding) parameters.

    The optics, sensor and wavelength grid are those of Config; this adds the
    camera, the noise model and the estimator settings.

    Noise level. One frame collects `photons_per_frame` photoelectrons in
    total (all wavelengths, before the band-limit loss), which puts the
    brightest 20 um pixel at ~4.5e4 e-, below a typical InGaAs full well.
    `n_frames` frames are summed, so the measurement has
    n_frames * photons_per_frame photoelectrons and a read-noise variance of
    n_frames * read_noise_e**2 per pixel.
    """
    # Camera
    camera_pitch: float = 20e-6       # camera pixel pitch [m]; multiple of the grid dx
    roi_pixels: int = 48              # ROI side in camera pixels, centred on the +1 order

    # Noise (Eq. 9)
    photons_per_frame: float = 1e7    # photoelectrons per frame at zero shift
    read_noise_e: float = 100.0       # read noise per pixel per frame [e- rms]
    n_frames: int = 100               # frames summed per measurement

    # Estimator
    delta_gammas: float = 0.02        # central-difference step delta / gamma (Eq. 7)
    Sn_nm_per_riu: float = 200.0      # assumed bulk sensitivity S_n [nm/RIU] (Eq. 1, 12)
    gn_iterations: int = 10           # Gauss-Newton re-linearizations

    # Monte Carlo (Section 5)
    test_shifts_gammas: tuple = (-0.1, -0.05, 0.03, 0.07, 0.1)
    mc_trials: int = 2000
    seed: int = 2

    @property
    def photons(self):
        """Total photoelectrons per measurement at zero shift."""
        return self.photons_per_frame * self.n_frames

    @property
    def read_var(self):
        """Read-noise variance per pixel per measurement [e-^2]."""
        return self.n_frames * self.read_noise_e ** 2


E2 = Embodiment2Config()
