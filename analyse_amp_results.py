import glob
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import scipy as sp
import pandas as pd

event_name = "GW200302"

path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/run2/weights_{}_*_IMRPhenomXPHM.csv".format(event_name))
s_path_list = sorted(path_list)
labels = []
bf_list = []
eff_list = []

for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    
    text = file_name.split('_IMR')
    text2 = text[0].split('=')
    label = text2[1]
    
    # if float(label)>150.0:
    #     continue
    
    labels.append(float(label))
    bf = np.nansum(data)/len(data)
    eff = (np.sum(data))**2 /np.sum(np.square(data))/len(data)
    bf_list.append(bf)
    eff_list.append(eff)

path = "/home/shunyin.cheung/memory_GWTC3/run2/weights_{}_IMRPhenomXPHM.csv".format(event_name)
data = np.genfromtxt(path)
bf = np.nansum(data)/len(data)
# labels2 = [1.0] + labels
# bf_list2 = [bf] + bf_list

# parallel sort both arrays so that np.trapz works properly.
s_labels, s_bf_list, s_eff_list = (list(t) for t in zip(*sorted(zip(labels, bf_list, eff_list)))) 

print('efficiency', s_eff_list)
print('bf array: ',s_bf_list)
print('amplitudes: ',s_labels)

ln_labels = np.log(np.array(s_labels))
s_bf_list = np.array(s_bf_list)
s_eff_list = np.array(s_eff_list)*100


area = np.trapz(s_bf_list, s_labels)

prob = 1/area * s_bf_list

# to calculate highest posterior density interval, I first interpolate my graph.

prob_int = sp.interpolate.interp1d(s_labels, prob)

amplitudes = np.arange(0.1, 50, 0.1)

new_prob = prob_int(amplitudes)

# then I use cdf distribution.

cdf_norm = np.sum(new_prob)
cdf = np.cumsum(new_prob)/cdf_norm

# find when cdf equals 0.9

for i, value in enumerate(cdf):
    if cdf[i] > 0.9:
        key = amplitudes[i]
        break


# plt.figure()
# plt.plot(amplitudes, new_prob)
# plt.savefig('tests/pdf_test.png')

# plt.figure()
# plt.plot(amplitudes, cdf)
# plt.savefig('tests/cdf_test.png')

plt.figure()
plt.title(event_name)
plt.plot(s_labels, prob, linestyle='None', marker='o')
plt.fill_between(s_labels, prob, color='cornflowerblue', alpha=0.5)
# plt.axvline(0, linestyle='dashed', color='black')
# plt.axvline(key, linestyle='dashed', color='black')
#plt.xscale('log')
plt.xlabel(f'A', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, np.max(s_labels))
plt.ylim(0, np.max(prob)+ 0.1*np.max(prob))
plt.tight_layout()
plt.savefig('results/{}_amplitude_vs_posterior.pdf'.format(event_name))
plt.savefig('results/{}_amplitude_vs_posterior.png'.format(event_name))

plt.figure()
plt.title(event_name)
plt.plot(s_labels, prob, linestyle='None', marker='o')
plt.fill_between(s_labels, prob, color='cornflowerblue', alpha=0.5)
plt.xlabel(f'A', fontsize=18)
plt.xscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(np.min(s_labels), np.max(s_labels))
plt.ylim(0, np.max(prob)+ 0.1*np.max(prob))
plt.tight_layout()
plt.savefig('results/{}_log10_amplitude_vs_posterior.pdf'.format(event_name))
plt.savefig('results/{}_log10_amplitude_vs_posterior.png'.format(event_name))


s_labels = np.array(s_labels)
s_bf_list = np.array(s_bf_list)
result = np.stack((s_labels, s_bf_list), axis=1)
np.savetxt("results/{}_amplitude_posterior_results.csv".format(event_name), result, delimiter=',')


csvdata = open('results/{}_memory_snr_vs_amp.csv'.format(event_name))
memory_snr_table = np.loadtxt(csvdata)
print(memory_snr_table)
memory_amp = memory_snr_table[:,0]
memory_snr = memory_snr_table[:,1]

fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx() 
plt.title('{}'.format(event_name))
lns1 = ax1.plot(s_labels, s_bf_list, label='BF')
lns2 = ax2.plot(memory_amp, memory_snr, label='memory SNR', color='orange')
ax1.set_xlabel('A')
ax1.set_ylabel('BF')
ax1.set_xlim(0, np.max(s_labels))
ax2.set_ylabel('optimal SNR')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.savefig('results/{}_memory_snr_vs_amp.png'.format(event_name))
plt.savefig('results/{}_memory_snr_vs_amp.png'.format(event_name))

fig, axs = plt.subplots(3, figsize=(9, 6))
axs[0].plot(s_labels, s_bf_list)
axs[0].set_title('{}'.format(event_name))
axs[0].set_ylabel('Bayes factor')
axs[1].plot(memory_amp, memory_snr)
axs[1].set_ylabel('memory SNR')
axs[2].plot(s_labels, s_eff_list)
axs[2].set_ylabel('efficiency (%)')
axs[2].set_xlabel('amplitude')
axs[0].label_outer()
axs[1].label_outer()
axs[2].label_outer()
axs[0].set_ylim(0, np.max(s_bf_list))
axs[1].set_ylim(0, np.max(memory_snr))
axs[2].set_ylim(0, 100)
for ax in axs:
    ax.set_xlim(0, 128)

print(event_name)
plt.savefig('results/{}_three_metric_plot.png'.format(event_name))