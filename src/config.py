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
