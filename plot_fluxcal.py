import sncosmo
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    direc = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SNDATA_ROOT/SIM/QC_test_AGN'
    filename = 'QC_test_AGN_SN000008.DAT'
    direc_save = '~/flux_plots'
    os.makedirs(direc_save, exist_ok=True)

    meta, tables = sncosmo.read_snana_ascii(os.path.join(direc, filename), default_tablename='OBS')
    # table_name: name before each observation  QC_test_AGN_SN000003.DAT  QC_test_BAYESN_SN000001.DAT
    #plt.plot(tables['OBS']['MJD'], tables['OBS']['FLUXCAL'])
    colors = {'u':'purple', 'g':'green', 'r':'red', 'i':'blue', 'z': 'c', 'Y': 'yellow'}
    plt.errorbar(tables['OBS']['MJD'], tables['OBS']['FLUXCAL'], yerr=tables['OBS']['FLUXCALERR'], fmt="o")

    plt.errorbar(tables['OBS']['MJD'], 27.5 - 2.5 * np.log10(tables['OBS']['FLUXCAL']),
                 yerr=- 2.5 * np.log10(tables['OBS']['FLUXCAL'])/(np.log(10) *tables['OBS']['FLUXCALERR']), fmt="o")

    plt.savefig(os.path.join(direc_save, f'{filename}.png'))

if __name__ == '__main__':
    main()



