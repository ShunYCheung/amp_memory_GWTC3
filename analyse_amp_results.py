import glob
import numpy as np
import matplotlib.pyplot as plt

event_name = "GW190412"

path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/run2/weights_{}_*_IMRPhenomXPHM.csv".format(event_name))
s_path_list = sorted(path_list)
labels = []
bf_list = []
print(s_path_list)

for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    print('data', data)
    
    text = file_name.split('_IMR')
    text2 = text[0].split('=')
    #print(text2)
    label = text2[1]
    
    if float(label)>100.0:
        continue
    
    labels.append(float(label))
    bf = np.nansum(data)/len(data)
    #print('bf', bf)
    bf_list.append(bf)
    
path = "/home/shunyin.cheung/memory_GWTC3/run2/weights_{}_IMRPhenomXPHM.csv".format(event_name)
data = np.genfromtxt(path)
bf = np.nansum(data)/len(data)
labels2 = [1.0] + labels
bf_list2 = [bf] + bf_list

# parallel sort both arrays so that np.trapz works properly.
s_labels, s_bf_list = (list(t) for t in zip(*sorted(zip(labels2, bf_list2)))) 

print('bf array: ',s_bf_list)
print('amplitudes: ',s_labels)

ln_labels = np.log(np.array(s_labels))
s_bf_list = np.array(s_bf_list)

area = np.trapz(s_bf_list, s_labels)

prob = 1/area * s_bf_list

plt.figure()
plt.title(event_name)
plt.plot(s_labels, prob, linestyle='None', marker='o')
plt.fill_between(s_labels, prob, color='cornflowerblue', alpha=0.5)
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
