# test of the class by adding ERDF

import numpy as np
from astropy import constants, units
from astropy.cosmology import Planck18
from scipy import integrate
from scipy.interpolate import UnivariateSpline

M_sun = constants.M_sun.cgs.value
c = constants.c.cgs.value
pc = (1 * units.pc).to_value(units.cm)
G = constants.G.cgs.value
sigma_sb = constants.sigma_sb.cgs.value
h = constants.h.cgs.value
k_B = constants.k_B.cgs.value


class Spline:
    def __init__(self, x, y, sampling_size=1, rng=None):
        self.x = x
        self.y = y
        self.rng = np.random.default_rng(rng)

        self.inv_cdf_func = self.inv_cdf()
        self.inv_samples = self.inv_trans_sampling(sampling_size=sampling_size)

    def inv_cdf(self):
        # y(x); y is the CDF of x
        # return inv_cdf function
        x = self.x
        y = self.y
        pdf_spline = UnivariateSpline(x=x, y=y, s=0, k=3)
        cdf_spline = pdf_spline.antiderivative()
        assert cdf_spline(x[0]) == 0
        norm = cdf_spline(x[-1])
        inv_cdf_spline = UnivariateSpline(x=cdf_spline(x) / norm, y=x, s=0, k=3)

        return inv_cdf_spline

    def inv_trans_sampling(self, sampling_size=1):
        # input inverse cdf function.
        r = self.rng.random(int(sampling_size))
        return self.inv_cdf_func(r)


class AGN:
    def __init__(self, t0: float, Mi: float, M_BH: float, lam: np.ndarray, edd_ratio: float, rng):  # constructor
        self.lam = np.asarray(lam)
        self.t0 = t0
        self.rng = np.random.default_rng(rng)

        #         self.Fnu_average = self.find_Fnu_average(self.lam, M_BH, None)
        self.ME_dot = self.find_ME_dot(M_BH)
        self.MBH_dot = self.find_MBH_dot(self.ME_dot, M_BH, edd_ratio)
        self.Fnu_average = self.find_Fnu_average_standard_disk(self.MBH_dot, self.lam, M_BH)

        self.tau = self.find_tau_v(self.lam, Mi, M_BH)
        self.sf_inf = self.find_sf_inf(self.lam, Mi, M_BH)

        self.t = t0
        self.delta_m = self._random() * self.sf_inf

    def step(self, t):
        dt = t - self.t
        self.t = t

        self.delta_m = (
                self.delta_m * np.exp(-dt / self.tau)
                + self.sf_inf * np.sqrt(1 - np.exp(-2 * dt / self.tau)) * self._random()
        )

    def __call__(self, t):
        self.step(t)
        return self.Fnu

    @property
    def Fnu(self):
        return 10 ** (-0.4 * self.delta_m) * self.Fnu_average

    def _random(self):
        return self.rng.normal(size=self.lam.size)

    @staticmethod
    def find_ME_dot(M_BH):
        # return in g/s
        return 1.4e18 * M_BH / M_sun

    @staticmethod
    def find_MBH_dot(ME_dot, M_BH, eddington_ratio):
        return ME_dot * eddington_ratio

    @staticmethod
    def T_0(M, Mdot, r_in):
        return (2 ** (3 / 4) * (3 / 7) ** (7 / 4) * (G * M * Mdot / (np.pi * sigma_sb * r_in ** 3)) ** (1 / 4))

    @staticmethod
    def r_0(r_in):
        return ((7 / 6) ** 2 * r_in)

    @staticmethod
    def x_fun(nu, T0, r, r0):
        return (h * nu / (k_B * T0) * (r / r0) ** (3 / 4))

    def find_flux_standard_disk(self, Mdot, nu, rin, rout, i, d, M):
        T0 = self.T_0(M, Mdot, rin)
        r0 = self.r_0(rin)
        xin = self.x_fun(nu, T0, rin, r0)
        xout = self.x_fun(nu, T0, rout, r0)
        fun_integr = lambda x: (x ** (5 / 3)) / (np.exp(x) - 1)
        #     integ, inte_err = integrate.quad(fun_integr, xin, xout)
        integ, inte_err = integrate.quad(fun_integr, 0, np.inf)

        return ((16 * np.pi) / (3 * d ** 2) * np.cos(i) * (k_B * T0 / h) ** (8 / 3) * h * (nu ** (1 / 3)) / (c ** 2) * (
                r0 ** 2) * integ)

    def find_Fnu_average_standard_disk(self, MBH_dot, lam, M_BH):
        flux_av = self.find_flux_standard_disk(MBH_dot, c / lam, rin=1, rout=1, i=0, d=10 * pc,
                                               M=M_BH)  # do we need i, d?
        return flux_av

    @staticmethod
    def find_Fnu_average(lam, M_BH, eddington_ratio):
        # Input wavelength as array.
        # Return baseline (average value).

        z = 0.2
        mu = Planck18.distmod(z).value
        # return 1e-29 * (lam / 5000e-8)**(-1/3)
        F_av = 10 ** (-0.4 * (20 + 48.6 - mu))
        Fnu_ave = np.full_like(lam, F_av)
        return Fnu_ave

    @staticmethod
    def find_tau_v(lam, Mi=-23, M_BH=1e9 * M_sun):
        """Input frequency v in Hz, i band magnitude (default is -23), Black whole mass in g (defalt is 10^9 solar mass).

        Return timescale in s."""

        A = 2.4  # self.rng.normal(2.4, ...)
        B = 0.17
        C = 0.03
        D = 0.21
        # add C, D, BH_mass, Mi
        return 10 ** (A + B * np.log10(lam / (4000e-8))
                      + C * (Mi + 23) + D * np.log10(M_BH / (1e9 * M_sun)))  # e-8 angstrom

    @staticmethod
    def find_sf_inf(lam, Mi=-23, M_BH=1e9 * M_sun):
        """Input frequency in Hz, i band magnitude (default is -23), Black whole mass in g (defalt is 10^9 solar mass).

        Return Structure Function at infinity in mag."""

        A = -0.51
        B = -0.479
        C = 0.13
        D = 0.18

        return 10 ** (A + B * np.log10(lam / (4000e-8))
                      + C * (Mi + 23) + D * np.log10(M_BH / (1e9 * M_sun)))


class gensed_BAYESN:
    def __init__(self, PATH_VERSION, OPTMASK, ARGLIST, HOST_PARAM_NAMES):
        print('__init__', flush=True)
        self.agn = None
        print(1, flush=True)
        # self.host_param_names = [x.upper() for x in HOST_PARAM_NAMES.split(',')]
        # self.PATH_VERSION = os.path.expandvars(PATH_VERSION)

        # Check how to get seed from SNANA
        self.rng = np.random.default_rng(0)
        print(2, flush=True)

        self.wavelen = 100
        print(3, flush=True)
        self.wave = np.logspace(np.log10(100e-8), np.log10(20000e-8), self.wavelen)
        print('__init__', flush=True)

        self.edd_ratio = None  # ??

        self.log_lambda_min = -8.5
        self.log_lambda_max = 0.5
        self.nbins = 1000

    @staticmethod
    def ERDF(lambda_Edd, galaxy_type='Blue', rng=None):
        """
        ERDF for blue galaxies (radiatively-efficient, less massive)
        """
        rng = np.random.default_rng(rng)

        # Lbr = 10**38.1 lambda_br M_BH_br
        # 10^41.67 = 10^38.1 * 10^x * 10^10.66
        if galaxy_type == 'Red':
            xi = 10 ** -2.13
            lambda_br = 10 ** rng.normal(-2.81, np.mean([0.22, 0.14]))
            delta1 = rng.normal(0.41 - 0.7, np.mean([0.02, 0.02]))  # > -0.45 won't affect LF
            delta2 = rng.normal(1.22, np.mean([0.19, 0.13]))

        if galaxy_type == 'Blue':
            xi = 10 ** -1.65
            lambda_br = 10 ** rng.normal(-1.84, np.mean([0.30, 0.37]))
            delta1 = rng.normal(0.471 - 0.7, np.mean([0.02, 0.02]))  # > -0.45 won't affect LF
            delta2 = rng.normal(2.53, np.mean([0.68, 0.38]))

        # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
        # What sets the break? Transfer from radiatively efficient to inefficient accretion?

        # parameters from this paper:
        # https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf
        #     print('(lambda_Edd/lambda_br)**delta1:\n',(lambda_Edd/lambda_br)**delta1)

        return xi * ((lambda_Edd / lambda_br) ** delta1 + (lambda_Edd / lambda_br) ** delta2) ** -1

    def fetchSED_NLAM(self):
        """
        Returns the length of the wavelength vector
        """
        print('fetchSED_NLAM', flush=True)
        return self.wavelen

    def fetchSED_LAM(self):
        """
        Returns the wavelength vector
        """
        print('fetchSED_LAM', flush=True)
        wave_aa = self.wave * 1e8
        return wave_aa.tolist()

    def fetchSED_BAYESN(self, trest, maxlam=5000, external_id=1, new_event=1, hostparams=''):
        print('fetchSED_BAYESN', flush=True)
        if new_event:
            lambda_ = np.logspace(self.log_lambda_min, self.log_lambda_max, self.nbins + 1)
            xi_blue = self.ERDF(lambda_Edd=lambda_, rng=self.rng)
            ERDF_spline = Spline(lambda_, xi_blue, sampling_size=1, rng=self.rng)
            self.edd_ratio = ERDF_spline.inv_samples

            self.agn = AGN(t0=trest, Mi=-23, M_BH=1e9 * M_sun, lam=self.wave, edd_ratio=self.edd_ratio, rng=self.rng)

        else:
            self.agn.step(trest)
        print('fetchSED_BAYESN', flush=True)
        Flambda = self.agn.Fnu * c / self.wave ** 2 * 1e-8
        return Flambda.tolist()

    def fetchParNames_BAYESN(self):
        print('fetchParNames_BAYESN', flush=True)
        return []

    def fetchNParNames_BAYESN(self):
        print('fetchNParNames_BAYESN', flush=True)
        return 0

    def fetchParVals_BAYESN_4SNANA(self, varname):
        print('fetchParVals_BAYESN_4SNANA', flush=True)
        return 'SNANA'
