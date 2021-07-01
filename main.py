import pathlib
from fiona import env
import pyhydrophone as pyhy
import argparse

from soundexplorer import dataset, shipwreck


parser = argparse.ArgumentParser(description='Generate a dataset and some plots for data exploration in the bpns')
parser.add_argument('summary_path', metavar='N', type=str, nargs='+', help='csv file with a list of all the ')
args = parser.parse_args()

# Acoustic Data
summary_path = pathlib.Path(args.summary_path[0])
include_dirs = False

# Output folder
output_folder = summary_path.parent.joinpath('data_exploration')

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)

upam_model = 'uPam'
upam_name = 'Seiche'
upam_serial_number = 'SM7213'
upam_sensitivity = -196.0
upam_preamp_gain = 0.0
upam_Vpp = 20.0
upam = pyhy.Seiche(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity,
                   preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)

aural_name = 'Aural'
aural_model = 'M2'
aural_serial_number = 0
aural_sensitivity = -164.0
aural_preamp_gain = 16.0
aural_Vpp = 2.0
aural = pyhy.MTE(name=aural_name, model=aural_model, serial_number=aural_serial_number, sensitivity=aural_sensitivity,
                 preamp_gain=aural_preamp_gain, Vpp=aural_Vpp, string_format='%y%m%d_%H%M%S')


instruments = {'SoundTrap': soundtrap, 'uPam': upam, 'B&K': bk, 'Aural': aural}

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
band_lf = [50, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]
band_list = [band_lf, band_mf, band_hf]
features = ['rms', 'sel', 'aci']
third_octaves = None

env_vars = ['spatial_data', 'sea_state', 'time_data', 'sea_bottom', 'shipping', 'shipwreck']

if __name__ == "__main__":
    dataset = dataset.DataSet(summary_path, output_folder, instruments, features, third_octaves, band_list, binsize, nfft)
    dataset.generate_entire_dataset(env_vars=env_vars)
    # dataset.read_all_deployments()
    # dataset.read_dataset()
    # dataset.plot_all_features_evo()
    # dataset.plot_all_features_distr()



