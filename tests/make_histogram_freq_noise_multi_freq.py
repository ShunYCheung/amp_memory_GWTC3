import numpy as np
import bilby
import glob
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2

ifo_list = bilby.gw.detector.InterferometerList([])

det= 'H1'
freq_list = [20, 40, 80, 160, 320, 640]

event_list = [f'run{i}' for i in range(100)]

logger = bilby.core.utils.logger

data_point_list = np.array([])

# define interferometer objects
for event_name in event_list:   
    data_path = glob.glob(f'/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/{event_name}_*_{det}.txt')
    psd_path = glob.glob(f'/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/{event_name}_*_{det}_psd.dat')
    #print(f'calling data from {data_path}')
    #print(f'calling psd from {psd_path}')
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.read(data_path[0])
    ifo.strain_data.set_from_gwpy_timeseries(data)
    
    psd = np.genfromtxt(psd_path[0])
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=psd[:, 0], psd_array=psd[:, 1]
    )
    data_list = np.array([])
    for freq in freq_list:
        data_at_freq = ifo.frequency_domain_strain[ifo.frequency_array == freq]
        data_list = np.append(data_list, np.real(data_at_freq))
    print(data_list)

    print(data_point_list.shape)
    print(data_list.shape)

    if len(data_point_list) == 0:
        data_point_list = data_list
    else:
        data_point_list = np.vstack((data_point_list, data_list))

print(data_point_list)
# Sample data (replace with your data)
reduced_chi_sq_list = np.array([])

for i in range(len(freq_list)):
    data = data_point_list[:, i]

    # Step 1: Create a histogram
    hist_data, bin_edges = np.histogram(data, bins=10, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 2: Define a Gaussian model function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    # Step 3: Perform least squares fitting
    initial_guess = [32, np.mean(data), np.std(data)]
    params, pcov = curve_fit(gaussian, bin_centers, hist_data, p0=initial_guess)

    perr = np.sqrt(np.diag(pcov))

    print('Errors in parameters',perr)

    # Extracting fitted parameters
    amplitude, mean, stddev = params

    # Step 1: Calculate expected frequencies using the fitted model
    expected_frequencies = gaussian(bin_centers, *params)
    print(expected_frequencies)
    print(hist_data)
    # Recalculate expected frequencies to ensure the sum matches the observed frequencies
    total_observed = np.sum(hist_data)
    expected_frequencies_normalized = expected_frequencies * (total_observed / np.sum(expected_frequencies))

    # Now use the normalized expected frequencies in the chi-squared test
    chi2_statistic, p_value = chisquare(hist_data, f_exp=expected_frequencies_normalized)
    #chi2_statistic2 = np.sum(((hist_data - expected_frequencies) ** 2) / expected_frequencies)

    # Step 3: Calculate degrees of freedom
    degrees_of_freedom = len(bin_centers) - len(params)
    reduced_chi_sq = chi2_statistic/degrees_of_freedom

    reduced_chi_sq_list = np.append(reduced_chi_sq_list, reduced_chi_sq)

plt.figure()
plt.plot(freq_list, reduced_chi_sq_list)
plt.yscale('log')
plt.xlabel('frequency (Hz)')
plt.ylabel(f'reduced $\chi^2$')
plt.savefig('test_results/gaussian_chi_squared_vs_freq.png')
