import numpy as np
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from gensed_base import gensed_base

M_sun = constants.M_sun.cgs.value
c = constants.c.cgs.value
pc = (1 * units.pc).to_value(units.cm)
G = constants.G.cgs.value
sigma_sb = constants.sigma_sb.cgs.value
h = constants.h.cgs.value
k_B = constants.k_B.cgs.value


class DistributionSampler:
    """
    A class to build a sampler by inverse sampling
    """
    def __init__(self, x, y, rng=None):
        # input y and x, which y is CDF of x
        self.rng = np.random.default_rng(rng)
        self.inv_cdf_spline = self.inv_cdf(x, y)

    @staticmethod
    def inv_cdf(x, y):
        # y(x); y is the CDF of x
        # return inverse cdf function
        pdf_spline = UnivariateSpline(x=x, y=y, s=0, k=3)
        cdf_spline = pdf_spline.antiderivative()
        assert cdf_spline(x[0]) == 0
        norm = cdf_spline(x[-1])
        inv_cdf_spline = UnivariateSpline(x=cdf_spline(x) / norm, y=x, s=0, k=3)

        return inv_cdf_spline

    def inv_trans_sampling(self, sampling_size=1):
        # input: inverse cdf function.
        # output: random samples
        r = self.rng.random(int(sampling_size))
        return self.inv_cdf_spline(r)


class AGN:
    def __init__(self, t0: float, Mi: float, M_BH: float, lam: np.ndarray, edd_ratio: float, rng):
        self.lam = np.asarray(lam)
        self.t0 = t0
        self.rng = np.random.default_rng(rng)

        self.ME_dot = self.find_ME_dot(M_BH)
        self.MBH_dot = self.find_MBH_dot(self.ME_dot, M_BH, edd_ratio)
        self.Fnu_average = 2 * self.find_Fnu_average_standard_disk(self.MBH_dot, self.lam, M_BH) # quick fix to double the baseline Fnu

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
        # Correct Fnu with its baseline (average value)
        # Output: Fnu: spectral density along the frequency axes
        return 10 ** (-0.4 * self.delta_m) * self.Fnu_average

    def _random(self):
        return self.rng.normal(size=self.lam.size)

    @staticmethod
    def find_ME_dot(M_BH):
        # Input: BH mass in g
        # return: Accretion rate at Eddington luminosity in g/s
        return 1.4e18 * M_BH / M_sun

    @staticmethod
    def find_MBH_dot(ME_dot, M_BH, eddington_ratio):
        # Input:
        # ME_dot: Accretion rate at Eddington luminosity in g/s
        # eddington ratio
        return ME_dot * eddington_ratio

    @staticmethod
    def T_0(M, Mdot, r_in):
        # calculate T0 based on standard disk model
        # input:
        # M: mass of the gravitating centre
        # Mdot: accretion rate at previous time step
        # r_in: the inner radius of the accretion disc
        # output:
        # T_0: Effective temperature at r0. Same as the maximum effective temperature at the disc surface (Tmax)
        return (2 ** (3 / 4) * (3 / 7) ** (7 / 4) * (G * M * Mdot / (np.pi * sigma_sb * r_in ** 3)) ** (1 / 4))

    @staticmethod
    def r_0(r_in):
        # return r0 based on standard disk model
        # input: r_in: the inner radius of the accretion disc
        # output: r0:  the initial radius of the ring
        return ((7 / 6) ** 2 * r_in)

    @staticmethod
    def x_fun(nu, T0, r, r0):
        """
        calculate variable of integration x
        :param nu: frequency
        :param T0: Effective temperature at r0. Same as the maximum effective temperature at the disc surface (Tmax)
        :param r: radius of the accretion disc
        :param r0: the initial radius of the ring
        :return: variable of integration x
        """
        return (h * nu / (k_B * T0) * (r / r0) ** (3 / 4))

    def find_flux_standard_disk(self, Mdot, nu, rin, rout, i, d, M):
        """
        function to calculate flux based on standard disk model
        Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        https://link.springer.com/chapter/10.1007/978-3-319-93009-1_1
        input:
        Mdot: accretion rate at previous time step
        nu: frequency
        rin: the inner radius of the accretion disc
        i: inclination
        d: distance
        M: mass of the gravitating centre
        output:
        flux at given time step
        """
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
        """
        function to return average flux in purpose of normalization
        expect 20 mag in r band for observer at z = 0.2 (z from .input file)
        Input:
        lam: wavelength array.
        Output:
        Return baseline (average value).
        """

        z = 0.2
        mu = Planck18.distmod(z).value
        F_av = 10 ** (-0.4 * (20 + 48.6 - mu))
        Fnu_ave = np.full_like(lam, F_av)
        return Fnu_ave

    @staticmethod
    def find_tau_v(lam, Mi=-23, M_BH=1e9 * M_sun):
        """
        Return timescale for DRW model.
        equation and parameters for A, B, C, D adopted from Suberlak et al. 2021

        Input frequency v in Hz, i band magnitude (default is -23), Black hole mass in g (defalt is 10^9 solar mass).
        Return timescale in s.
        """

        A = 2.4  # self.rng.normal(2.4, ...)
        B = 0.17
        C = 0.03
        D = 0.21
        # add C, D, BH_mass, Mi
        return 10 ** (A + B * np.log10(lam / (4000e-8))
                      + C * (Mi + 23) + D * np.log10(M_BH / (1e9 * M_sun)))  # e-8 angstrom

    @staticmethod
    def find_sf_inf(lam, Mi=-23, M_BH=1e9 * M_sun):
        """
        Input frequency in Hz, i band magnitude (default is -23), Black hole mass in g (defalt is 10^9 solar mass).
        equation and parameters for A, B, C, D adopted from Suberlak et al. 2021

        Return Structure Function at infinity in mag.
        """

        A = -0.51
        B = -0.479
        C = 0.13
        D = 0.18

        return 10 ** (A + B * np.log10(lam / (4000e-8))
                      + C * (Mi + 23) + D * np.log10(M_BH / (1e9 * M_sun)))


class gensed_AGN(gensed_base):
    # round input rest time by 10**_trest_digits days
    _trest_digits = 8

    def __init__(self, PATH_VERSION, OPTMASK, ARGLIST, HOST_PARAM_NAMES):
        print('__init__', flush=True)
        self.agn = None
        self.trest = None
        self.sed = None
        self.sed_Fnu = None
        print(1, flush=True)

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

        self.M_BH = None
        self.Mi = None

        lambda_ = np.logspace(self.log_lambda_min, self.log_lambda_max, self.nbins + 1)
        xi_blue = self.ERDF(lambda_Edd=lambda_, rng=self.rng)
        self.ERDF_spline = DistributionSampler(lambda_, xi_blue, rng=self.rng)

    def _get_Flambda(self):
        return self.agn.Fnu * c / self.wave ** 2 * 1e-8

    @staticmethod
    def ERDF(lambda_Edd, galaxy_type='Blue', rng=None):
        """
        ERDF for blue galaxies (radiatively-efficient, less massive)
        https://github.com/burke86/imbh_forecast/blob/master/var.ipynb

        https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
        parameters from this paper:
        https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf
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

        return xi * ((lambda_Edd / lambda_br) ** delta1 + (lambda_Edd / lambda_br) ** delta2) ** -1

    @staticmethod
    def M_BH_sample(rng):
        logMH_min = 7
        logMBH_max = 9
        return 10 ** (rng.uniform(logMH_min, logMBH_max))

    @staticmethod
    def find_L_bol(edd_ratio, M_BH):
        # Input: M_BH in unit of M_sun.
        # Return L_bol in erg/s
        return edd_ratio * 1.26e38 * M_BH

    @staticmethod
    def find_Mi(L_bol):
        # from Shen et al., 2013
        # https://adsabs.harvard.edu/full/2013BASI...41...61S
        return 90 - 2.5 * np.log10(L_bol)

    def fetchSED_NLAM(self):
        """
        Returns the length of the wavelength vector
        """
        print('fetchSED_NLAM', flush=True)
        return self.wavelen



    def prepEvent(self, trest, external_id, hostparams):
        # trest is sorted
        self.trest = np.round(trest, self._trest_digits)


        self.edd_ratio = self.ERDF_spline.inv_trans_sampling(sampling_size=1)

        self.M_BH = self.M_BH_sample(self.rng)  # M_BH in unit of M_sun
        L_bol = self.find_L_bol(self.edd_ratio, self.M_BH)  # L_bol: in erg/s
        self.Mi = self.find_Mi(L_bol)  # L_bol in erg/s

        print(f'M_BH:{self.M_BH}\n')
        print(f'L_bol: {L_bol}\n')
        print(f'Mi: {self.Mi}\n')
        print(f'Edd_ratio:{self.edd_ratio}')

        self.agn = AGN(t0=self.trest[0], Mi=self.Mi, M_BH=self.M_BH * M_sun, lam=self.wave, edd_ratio=self.edd_ratio,
                       rng=self.rng)

        # self.agn = AGN(t0=self.trest[0], Mi=-23, M_BH=1e9 * M_sun, lam=self.wave, edd_ratio=0.1, rng=self.rng)
        print(f'hostparams:{hostparams}')
        self.sed = {self.trest[0]: self._get_Flambda()}
        # TODO: consider a case of repeated t, we usually have several t = 0
        for t in self.trest[1:]:
            self.agn.step(t)
            self.sed[t] = self._get_Flambda()

    def test_AGN_flux(self, trest):
        self.trest = np.round(trest, self._trest_digits)
        self.agn = AGN(t0=self.trest[0], Mi=self.Mi, M_BH=self.M_BH * M_sun, lam=self.wave, edd_ratio=self.edd_ratio,
                       rng=self.rng)

        self.sed = {self.trest[0]: self._get_Flambda()}
        self.sed_Fnu = {self.trest[0]: self.agn.Fnu}
        # TODO: consider a case of repeated t, we usually have several t = 0
        for t in self.trest[1:]:
            self.agn.step(t)
            self.sed[t] = self._get_Flambda()
            self.sed_Fnu[t] = self.agn.Fnu


    def fetchSED_LAM(self):
        """
        Returns the wavelength vector
        """
        wave_aa = self.wave * 1e8
        # print('wave:',wave_aa)
        return wave_aa

    def fetchSED(self, trest, maxlam, external_id, new_event, hostparams):
        trest = round(trest, self._trest_digits)
        return self.sed[trest]

    def fetchParNames(self):
        return ['M_BH', 'Mi']

    def fetchParVals(self, varname):
        return getattr(self, varname)

def main():
    mySED = gensed_AGN('$SNDATA_ROOT/models/bayesn/BAYESN.M20',2,[],'z,AGE,ZCMB,METALLICITY')

    trest = np.arange(1, 1000, 0.1)

    #test for obj dbID=2534406
    L_bol = 10**(45.21136675358238)
    mySED.M_BH = 10 ** (8.564795254)
    mySED.edd_ratio = L_bol/ (1.26e38 * mySED.M_BH)
    mySED.rng = np.random.default_rng(0)
    mySED.Mi = mySED.find_Mi(L_bol)

    """
    #test for obj dbID=8442
    L_bol = 10**(46.61)
    mySED.M_BH = 10 ** 9.09
    mySED.edd_ratio = L_bol/ (1.26e38 * mySED.M_BH)
    mySED.rng = np.random.default_rng(0)
    mySED.Mi = mySED.find_Mi(L_bol)
    """
    """
    #default test
    mySED.Mi = -23
    mySED.M_BH = 1e9 # in unit of M_sun
    mySED.rng = np.random.default_rng(0)
    mySED.edd_ratio = 0.1
    """
    mySED.test_AGN_flux(trest)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)

    flux_firstWave = []
    sed_list = - 2.5 * np.log10(list(mySED.sed_Fnu.values())) - 48.5
    sed_list = sed_list + astropy.coordinates.Distance(z=0.805899978).distmod.value
    for i in range(len(mySED.sed_Fnu)):
        #flux_firstWave.append(sed_list[i][82])  #wave[82] = 8000 armstrong(i-band)
        #flux_firstWave.append(sed_list[i][54])  #wave[54] closest to 6156/(1+2.43) armstrong(corrected ps1 r band)
        flux_firstWave.append(sed_list[i][61])  #wave[61] closest to 4770/(1+ 0.805899978) armstrong(corrected SDSS g band)
    #print(list(my(sed_list.sed.values())[0])


    ax1.plot(trest*(1+0.805899978), flux_firstWave, 'g-')
    ax1.invert_yaxis()
    plt.show()
    fig.savefig('test_apparentmag_2534406.png')

    # def fetchSED_LAM(self):
    #     """
    #     Returns the wavelength vector
    #     """
    #     print('fetchSED_LAM', flush=True)
    #     wave_aa = self.wave * 1e8
    #     return wave_aa.tolist()

    # def fetchSED_BAYESN(self, trest, maxlam=5000, external_id=1, new_event=1, hostparams=''):
    #     print('fetchSED_BAYESN', flush=True)
    #     if new_event:
    #         lambda_ = np.logspace(self.log_lambda_min, self.log_lambda_max, self.nbins + 1)
    #         xi_blue = self.ERDF(lambda_Edd=lambda_, rng=self.rng)
    #         ERDF_spline = DistributionSampler(lambda_, xi_blue, rng=self.rng)
    #         self.edd_ratio = ERDF_spline.inv_trans_sampling(sampling_size=1)
    #
    #         self.M_BH = self.M_BH_sample(self.rng)
    #         L_bol = self.L_bol(self.edd_ratio, self.M_BH)  # L_bol: in erg/s
    #         self.Mi = self.Mi(L_bol)  # L_bol in erg/s
    #
    #         # self.agn = AGN(t0=trest, Mi=-23, M_BH=1e9 * M_sun, lam=self.wave, edd_ratio=self.edd_ratio, rng=self.rng)
    #         self.agn = AGN(t0=trest, Mi=self.Mi, M_BH=self.M_BH * M_sun, lam=self.wave, edd_ratio=self.edd_ratio, rng=self.rng)
    #
    #     else:
    #         self.agn.step(trest)
    #     print('fetchSED_BAYESN', flush=True)
    #     Flambda = self.agn.Fnu * c / self.wave ** 2 * 1e-8
    #     return Flambda.tolist()
    #
    # def fetchParNames_BAYESN(self):
    #     print('fetchParNames_BAYESN', flush=True)
    #     return []
    #
    # def fetchNParNames_BAYESN(self):
    #     print('fetchNParNames_BAYESN', flush=True)
    #     return 0
    #
    # def fetchParVals_BAYESN_4SNANA(self, varname):
    #     print('fetchParVals_BAYESN_4SNANA', flush=True)
    #     return 'SNANA'






if __name__ == '__main__':
    main()




