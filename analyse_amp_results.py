import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pathfinder import find_multiple_path


event_label = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 
                   'GW190929', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']


def make_results(event_name, path_list, outdir, label):
    
    s_path_list = sorted(path_list)
    amp_list = []
    bf_list = []
    eff_list = []

    for file_name in s_path_list:
        data = np.genfromtxt(file_name)

        bf = np.nansum(data)/len(data)
        eff = (np.nansum(data))**2 /np.nansum(np.square(data))/len(data)

        text = file_name.split('_IMR')
        text2 = text[0].split('=')
        amp = text2[1]
            
        amp_list.append(float(amp))
        bf_list.append(bf)
        eff_list.append(eff)
        

    # parallel sort both arrays so that np.trapz works properly.
    print(amp_list)
    print(bf_list)
    print(eff_list)
    s_amp_list, s_bf_list, s_eff_list = (list(t) for t in zip(*sorted(zip(amp_list, bf_list, eff_list)))) 

    s_bf_list = np.array(s_bf_list)
    s_eff_list = np.array(s_eff_list)
    
    s_eff_list = np.nan_to_num(s_eff_list)

    bf_int = sp.interpolate.interp1d(s_amp_list, s_bf_list)

    new_amp = np.linspace(np.min(s_amp_list), np.max(s_amp_list), 100000)

    new_bf = bf_int(new_amp)
    prob = new_bf/np.sum(new_bf)

    t = np.linspace(0, prob.max(), 100)
    integral = ((prob >= t[:, None]) * prob).sum(axis=1)

    f = sp.interpolate.interp1d(integral, t)
    probslevels = [0.9]
    t_contours = f(np.array(probslevels))
    contour = t_contours[0]
    idx = np.argwhere(np.diff(np.sign(prob - contour))).flatten()
    # print(idx)
    # if len(idx) < 2:
    #     idx = np.append(idx, [0])
    # elif len(idx) > 2:
    #     print('Error: credible interval has more than two values.')
    #     exit()
    
    ci90 = sorted(new_amp[idx])
    print('90 percent credible interval', ci90)

    plt.figure()
    plt.title(event_name)
    plt.axvline(1, linestyle='dashed', color='black')
    plt.fill_between(new_amp, prob, color='cornflowerblue', alpha=0.5)
    plt.xlabel(f'A', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, max(new_amp))
    plt.ylim(0, np.max(prob))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/{event_name}/{event_name}_amplitude_vs_posterior_{label}.pdf')
    plt.savefig(f'{outdir}/{event_name}/{event_name}_amplitude_vs_posterior_{label}.png')

    s_amp_list = np.array(s_amp_list)
    s_bf_list = np.array(s_bf_list)
    result = np.stack((s_amp_list, s_bf_list), axis=1)
    np.savetxt(f"{outdir}/{event_name}/{event_name}_amplitude_posterior_results_{label}.csv", result, delimiter=',')
    try:
        csvdata = open(f'{outdir}/{event_name}/{event_name}_memory_snr_vs_amp.csv')
        
    except:
        print(f'Error: memory snr file cannot be found for this event. Skip {event_name}')
        return None
    
    memory_snr_table = np.loadtxt(csvdata)
    memory_amp = memory_snr_table[:,0]
    memory_snr = memory_snr_table[:,1]

    fig, axs = plt.subplots(3, figsize=(9, 6))
    axs[0].plot(s_amp_list, s_bf_list)
    axs[0].set_title('{}'.format(event_name))
    axs[0].set_ylabel('Bayes factor')
    axs[1].plot(memory_amp, memory_snr)
    axs[1].set_ylabel('memory SNR')
    axs[2].plot(s_amp_list, s_eff_list)
    axs[2].set_ylabel('efficiency (%)')
    axs[2].set_xlabel('amplitude')
    axs[0].label_outer()
    axs[1].label_outer()
    axs[2].label_outer()
    axs[0].set_ylim(0, np.max(s_bf_list)+0.05*np.max(s_bf_list))
    axs[1].set_ylim(0, memory_snr[memory_amp == max(new_amp)])
    axs[2].set_ylim(0, np.max(s_eff_list))
    for ax in axs:
        ax.set_xlim(0, max(new_amp))

    plt.savefig(f'{outdir}/{event_name}/{event_name}_three_metric_plot_{label}.png')

    return None


def combine_posteriors(path_list, amplitudes, outdir):

    combined_bf = np.ones(len(amplitudes))
    count = 0 
    for path in path_list:
        try:
            data = np.genfromtxt(path, delimiter=',')
        except:
            print(f"Cannot find data file.")
            continue
        for i , amp in enumerate(data[:, 0]):
            #print(data[:, 0])
            if amp in amplitudes:
                combined_bf[np.where(amplitudes==amp)] *= data[i,1]
            else:
                print(f'missing data point A={amp} from {path}')
        count += 1

    print('Total number of events combined = ', count)

    bf_int = sp.interpolate.interp1d(amplitudes, combined_bf)

    new_amp = np.linspace(np.min(amplitudes), np.max(amplitudes), 100000)

    new_bf = bf_int(new_amp)
    prob = new_bf/np.sum(new_bf)

    t = np.linspace(0, prob.max(), 100)
    integral = ((prob >= t[:, None]) * prob).sum(axis=1)

    f = sp.interpolate.interp1d(integral, t)
    probslevels = [0.9]
    t_contours = f(np.array(probslevels))
    contour = t_contours[0]
    idx = np.argwhere(np.diff(np.sign(prob - contour))).flatten()

    # if len(idx) < 2:
    #     idx = np.append(idx, 0)
    # elif len(idx) > 2:
    #     print('Error: credible interval has more than two values.')
    #     exit()
    
    # ci90 = sorted(new_amp[idx])
    # x = sorted(idx)
    # print('90 percent credible interval', ci90)

    plt.figure()
    plt.fill_between(new_amp, prob, color='cornflowerblue', alpha=0.5)
    # plt.fill_between(new_amp[x[0]:x[1]], prob[x[0]:x[1]], color='cornflowerblue', alpha=0.9)
    plt.axvline(1, linestyle='dashed', color='black')
    plt.xlabel(f'A', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 100)
    plt.ylim(0, np.max(prob)+0.01*np.max(prob))
    plt.tight_layout()
    plt.savefig(f'{outdir}/combined_amplitude_posterior.pdf')
    plt.savefig(f'{outdir}/combined_amplitude_posterior.png')

    plt.figure()
    plt.fill_between(new_amp, prob, color='cornflowerblue', alpha=0.5)
    plt.xlabel(f'A', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 4)
    plt.ylim(0, np.max(prob)+0.01*np.max(prob))
    plt.tight_layout()
    plt.savefig(f'{outdir}/combined_amplitude_posterior_low_amp.pdf')
    plt.savefig(f'{outdir}/combined_amplitude_posterior_low_amp.png')


# result_dir = 'run4/GW170818/*XPHM.csv'
# outdir = 'results'
# label = '080124'
# path_list = find_multiple_path('GW170818', result_dir)
# make_results('GW170818', path_list, outdir, label)

for i in range(10):
    result_dir = f'memory_only_run/run{i}/*XPHM.csv'
    outdir = 'memory_only_run/results'
    label='memory_burst_120124'
    path_list = find_multiple_path(f'run{i}', result_dir)
    make_results(f'run{i}', path_list, outdir, label)