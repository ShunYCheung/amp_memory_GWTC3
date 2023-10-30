import glob
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import scipy as sp
import pandas as pd

event_label = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 
                   'GW190929', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']


def make_results(event_name):
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
        eff = (np.nansum(data))**2 /np.nansum(np.square(data))/len(data)
        bf_list.append(bf)
        eff_list.append(eff)

    path = "/home/shunyin.cheung/memory_GWTC3/run2/weights_{}_IMRPhenomXPHM.csv".format(event_name)
    data = np.genfromtxt(path)
    bf = np.nansum(data)/len(data)
    eff = (np.nansum(data))**2 /np.nansum(np.square(data))/len(data)

    labels = [1.0] + labels
    bf_list = [bf] + bf_list
    eff_list = [eff] + eff_list

    # parallel sort both arrays so that np.trapz works properly.
    s_labels, s_bf_list, s_eff_list = (list(t) for t in zip(*sorted(zip(labels, bf_list, eff_list)))) 

    ln_labels = np.log(np.array(s_labels))
    s_bf_list = np.array(s_bf_list)
    s_eff_list = np.array(s_eff_list)*100

    # area = np.trapz(s_bf_list, s_labels)
    # prob = 1/area * s_bf_list

    bf_int = sp.interpolate.interp1d(s_labels, bf)

    new_amp = np.linspace(0.0625, 128, 1000)

    new_bf = bf_int(new_amp)
    prob = new_bf/np.sum(new_bf)

    t = np.linspace(0, prob.max(), 100)
    integral = ((prob >= t[:, None]) * prob).sum(axis=1)

    f = sp.interpolate.interp1d(integral, t)
    probslevels = [0.9]
    t_contours = f(np.array(probslevels))
    contour = t_contours[0]
    idx = np.argwhere(np.diff(np.sign(prob - contour))).flatten()

    if len(idx) < 2:
        idx = np.append(idx, [0])
    elif len(idx) > 2:
        print('Error: credible interval has more than two values.')
        exit()
    
    ci90 = sorted(prob[idx])
    print('90 percent credible interval', ci90)

    plt.figure()
    plt.title(event_name)
    plt.plot(new_amp, prob, linestyle='None', marker='o')
    plt.fill_between(new_amp, prob, color='cornflowerblue', alpha=0.5)
    plt.axvline(ci90[0], linestyle='dashed', color='black')
    plt.axvline(ci90[1], linestyle='dashed', color='black')
    #plt.xscale('log')
    plt.xlabel(f'A', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, np.max(new_amp))
    plt.ylim(0, np.max(prob)+ 0.1*np.max(prob))
    plt.tight_layout()
    plt.savefig('results/{}_amplitude_vs_posterior.pdf'.format(event_name))
    plt.savefig('results/{}_amplitude_vs_posterior.png'.format(event_name))

    plt.figure()
    plt.title(event_name)
    plt.plot(new_amp, prob, linestyle='None', marker='o')
    plt.fill_between(new_amp, prob, color='cornflowerblue', alpha=0.5)
    plt.xlabel(f'A', fontsize=18)
    plt.xscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(np.min(new_amp), np.max(new_amp))
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
    axs[2].set_ylim(0, np.max(s_eff_list))
    for ax in axs:
        ax.set_xlim(0, np.max(s_labels))

    print(event_name)
    plt.savefig('results/{}_three_metric_plot.png'.format(event_name))
    return None

for event in event_label:
    make_results(event)