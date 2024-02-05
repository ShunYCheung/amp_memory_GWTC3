import numpy as np
import bilby
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2

ifo_list = bilby.gw.detector.InterferometerList([])

det= 'H1'
freq = 40.0

logger = bilby.core.utils.logger

data_point_list = []

sampling_frequency = 2048
duration = 4
start_time =0

# define interferometer objects
for i in range(1000): 
    ifo = bilby.gw.detector.get_empty_interferometer(det)  
    ifo.minimum_frequency = 20
    ifo.set_strain_data_from_power_spectral_density(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
    )

    data_at_freq = ifo.frequency_domain_strain[ifo.frequency_array == freq]
    data_point_list.append(np.real(data_at_freq)[0])


# Sample data (replace with your data)
data = data_point_list

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

# Step 4: Find the p-value (if not using chisquare directly)

print(f"Chi-squared statistic: {chi2_statistic}")
print(f"Reduced chi-squared statistic: {reduced_chi_sq}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {degrees_of_freedom}")

new_x = np.linspace(np.min(bin_centers), np.max(bin_centers), 100)

# Plot the histogram and the fitted curve
plt.title(f'Strain at {freq} Hz')
plt.hist(data, bins=10, density=False, alpha=0.6, label='Histogram')
#plt.plot(bin_centers, gaussian(bin_centers, *params), label='Fitted curve')
plt.plot(new_x, gaussian(new_x, *params), label=f'Gaussian')
plt.xlabel('Re(h)')
plt.legend()
plt.savefig(f'test_results/histogram_data_at_{freq}Hz_{det}_Gaussian_noise_least_squares_fit_n1000.png')

# Print the fitted parameters
print(f"Fitted parameters: amplitude = {amplitude}, mean = {mean}, stddev = {stddev}")

#mu, sigma = norm.fit(data_point_list)

# # Create the histogram

# print(data_point_list)
# plt.figure()
# plt.title(f'absolute strain at {freq}Hz')
# plt.hist(data_point_list)

# xmin, xmax = plt.xlim()
# ymin, ymax =plt.ylim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, sigma)
# plt.plot(x, p/max(p)*ymax, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma)

# plt.xlabel('Re(h)')
# plt.savefig(f'test_results/histogram_data_at_{freq}Hz_{det}.png')