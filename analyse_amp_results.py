import glob
import numpy as np
import matplotlib.pyplot as plt

path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/run1/run1_GW150914_weights_IMRPhenomXPHM*0.csv")
s_path_list = sorted(path_list)

labels = []
lnbf_list = []
"""
for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    
    lnbf = np.log(np.sum(data)/len(data))
    lnbf_list.append(lnbf)
    
    text = file_name.split('0.csv')
    text2 = text[0].split('=')
    #print(text2)
    label = text2[1]
    labels.append(float(label))

"""
for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    
    text = file_name.split('0.csv')
    text2 = text[0].split('=')
    #print(text2)
    label = text2[1]
    
    if float(label)>30.0:
        continue
    labels.append(float(label))
    lnbf = np.sum(data)/len(data)
    lnbf_list.append(lnbf)
    

labels2 = [1.0] + labels
lnbf_list2 = [np.exp(-0.167)] + lnbf_list

# parallel sort both arrays so that np.trapz works properly.
s_labels, s_lnbf_list = (list(t) for t in zip(*sorted(zip(labels2, lnbf_list2)))) 

#print(s_lnbf_list)
#print(s_labels)

ln_labels = np.log(np.array(s_labels))
s_lnbf_list = np.array(s_lnbf_list)
area = np.trapz(s_lnbf_list, ln_labels)

print(area)

prob = 1/area * s_lnbf_list

plt.figure()
#plt.title('GW150914')
plt.plot(ln_labels, prob, linestyle='None', marker='o')
plt.xlabel(f'ln(A)', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('GW150914_ln_amplitude_vs_posterior.pdf')
