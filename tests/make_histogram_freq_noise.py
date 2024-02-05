import numpy as np
import bilby
import glob
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2
from scipy.special import gamma


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def cauchy_pdf(x, A, x0, gamma):
    """
    Calculate the probability density function of a Cauchy distribution.

    Parameters:
    - x: The point(s) at which to evaluate the pdf.
    - x0: The location parameter (median of the distribution).
    - gamma: The scale parameter (half-width at half-maximum).

    Returns:
    - The value of the pdf at x.
    """
    return A / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))


def students_t_pdf(t, A, nu):
    """
    Calculate the probability density function of Student's t-distribution.

    Parameters:
    - t: The point(s) at which to evaluate the pdf.
    - nu: The degrees of freedom for the distribution.

    Returns:
    - The value of the pdf at t.
    """
    coefficient = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2))
    exponent = -(nu + 1) / 2
    return A*coefficient * (1 + t**2 / nu) ** exponent


ifo_list = bilby.gw.detector.InterferometerList([])

det= 'H1'
model = 'Gaussian'
freq = 20.0
print(f'Analysing freq {freq}Hz')

event_list = [f'run{i}' for i in range(100)]

logger = bilby.core.utils.logger

data_point_list = []

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

    data_at_freq = ifo.frequency_domain_strain[ifo.frequency_array == freq]
    data_point_list.append(np.real(data_at_freq)[0])

# Sample data
data = data_point_list

# Create a histogram
hist_data, bin_edges = np.histogram(data, bins=10, density=False)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.title(f'Strain at {freq} Hz')
plt.hist(data, bins=10, density=False, alpha=0.6, label='Histogram')

# Perform least squares fitting
if model == 'Gaussian':
    print('Using Gaussian distribution')
    initial_guess = [32, np.mean(data), np.std(data)]
    params, pcov = curve_fit(gaussian, bin_centers, hist_data, p0=initial_guess)

    amplitude, mean, stddev = params
    print(f"Fitted parameters: amplitude = {amplitude}, mean = {mean}, stddev = {stddev}")
    expected_frequencies = gaussian(bin_centers, *params)

    # Recalculate expected frequencies to ensure the sum matches the observed frequencies
    total_observed = np.sum(hist_data)
    expected_frequencies_normalized = expected_frequencies * (total_observed / np.sum(expected_frequencies))

    # Now use the normalized expected frequencies in the chi-squared test
    chi2_statistic, p_value = chisquare(hist_data, f_exp=expected_frequencies_normalized)

    degrees_of_freedom = len(bin_centers) - len(params)
    reduced_chi_sq = chi2_statistic/degrees_of_freedom

    new_x = np.linspace(np.min(bin_centers), np.max(bin_centers), 100)

    # Plot the histogram and the fitted curve
    #plt.plot(bin_centers, gaussian(bin_centers, *params), label='Fitted curve')
    plt.plot(new_x, gaussian(new_x, *params), label=f'Gaussian')

elif model == 'Cauchy':
    print('Using Caunchy distribution')
    initial_guess = [1e-21, np.median(data), 7e-23]
    bounds = ([0, -1e-22, 1e-25], [100, 1e-22, 1])
    params, pcov = curve_fit(cauchy_pdf, bin_centers, hist_data, p0=initial_guess, bounds=bounds)
    amplitude, x0, gammaa = params
    print(f"Fitted parameters: amplitude = {amplitude}, x0 = {x0}, gamma = {gammaa}")
    expected_frequencies = cauchy_pdf(bin_centers, *params)

    # Recalculate expected frequencies to ensure the sum matches the observed frequencies
    total_observed = np.sum(hist_data)
    expected_frequencies_normalized = expected_frequencies * (total_observed / np.sum(expected_frequencies))

    # Now use the normalized expected frequencies in the chi-squared test
    chi2_statistic, p_value = chisquare(hist_data, f_exp=expected_frequencies_normalized)

    degrees_of_freedom = len(bin_centers) - len(params)
    reduced_chi_sq = chi2_statistic/degrees_of_freedom

    new_x = np.linspace(np.min(bin_centers), np.max(bin_centers), 100)

    # Plot the histogram and the fitted curve
    #plt.plot(bin_centers, gaussian(bin_centers, *params), label='Fitted curve')
    plt.plot(new_x, cauchy_pdf(new_x, *params), label=f'Cauchy')

elif model == 'Student_t':
    print('Using student t distribution')
    initial_guess = [30, 1]
    params, pcov = curve_fit(students_t_pdf, bin_centers, hist_data, p0=initial_guess)

    amplitude, nu = params
    print(f"Fitted parameters: amplitude = {amplitude}, nu = {nu}")
    expected_frequencies = students_t_pdf(bin_centers, *params)
 
    # Recalculate expected frequencies to ensure the sum matches the observed frequencies
    total_observed = np.sum(hist_data)
    expected_frequencies_normalized = expected_frequencies * (total_observed / np.sum(expected_frequencies))

    # Now use the normalized expected frequencies in the chi-squared test
    chi2_statistic, p_value = chisquare(hist_data, f_exp=expected_frequencies_normalized)
    #chi2_statistic2 = np.sum(((hist_data - expected_frequencies) ** 2) / expected_frequencies)

    degrees_of_freedom = len(bin_centers) - len(params)
    reduced_chi_sq = chi2_statistic/degrees_of_freedom

    new_x = np.linspace(np.min(bin_centers), np.max(bin_centers), 100)

    # Plot the histogram and the fitted curve
    #plt.plot(bin_centers, gaussian(bin_centers, *params), label='Fitted curve')
    plt.plot(new_x, students_t_pdf(new_x, *params), label=f'Student t')


plt.xlabel('Re(h)')
plt.legend()
plt.savefig(f'test_results/histogram_data_at_{freq}Hz_{det}_least_squares_{model}_fit.png')

perr = np.sqrt(np.diag(pcov))
print('Errors in parameters',perr)
print(f"Chi-squared statistic: {chi2_statistic}")
print(f"Reduced chi-squared statistic: {reduced_chi_sq}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {degrees_of_freedom}")