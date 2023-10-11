import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/run2/weights_GW150914_IMRPhenomXPHM*0.csv")
s_path_list = sorted(path_list)

labels = []
lnbf_list = []

for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    
    text = file_name.split('0.csv')
    text2 = text[0].split('=')
    #print(text2)
    label = text2[1]
    
    if float(label)>100.0:
        continue
    labels.append(float(label))
    lnbf = np.sum(data)/len(data)
    lnbf_list.append(lnbf)
    
path = "/home/shunyin.cheung/memory_GWTC3/run2/weights_GW150914_IMRPhenomXPHM.csv"
data = np.genfromtxt(path)
lnbf = np.sum(data)/len(data)
labels2 = [1.0] + labels
lnbf_list2 = [lnbf] + lnbf_list

# parallel sort both arrays so that np.trapz works properly.
s_labels, s_lnbf_list = (list(t) for t in zip(*sorted(zip(labels2, lnbf_list2)))) 

print('lnbf array: ',s_lnbf_list)
print('amplitudes: ',s_labels)

ln_labels = np.log(np.array(s_labels))
s_lnbf_list = np.array(s_lnbf_list)

area = np.trapz(s_lnbf_list, s_labels)

prob = 1/area * s_lnbf_list

plt.figure()
plt.title('GW150914')
plt.plot(s_labels, prob, linestyle='None', marker='o')
plt.fill_between(s_labels, prob, color='cornflowerblue', alpha=0.5)
#plt.xscale('log')
plt.xlabel(f'A', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, np.max(s_labels))
plt.ylim(0, np.max(prob))
plt.tight_layout()
plt.savefig('GW150914_amplitude_vs_posterior_111023.pdf')

plt.figure()
plt.title('GW150914')
plt.plot(ln_labels, prob, linestyle='None', marker='o')
plt.fill_between(ln_labels, prob, color='cornflowerblue', alpha=0.5)
plt.xlabel(f'ln(A)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(np.min(ln_labels), np.max(ln_labels))
plt.ylim(0, np.max(prob))
plt.tight_layout()
plt.savefig('GW150914_ln_amplitude_vs_posterior_111023.pdf')
