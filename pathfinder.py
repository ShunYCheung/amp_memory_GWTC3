import glob, os


event_list = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 
                   'GW190929', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']


def find_path(event, dir):
    """
    Finds the result/data file that matches the specific event.
    Result file must be in the dir you specify.
    """
    path_list = glob.glob(dir)
    s_path_list = sorted(path_list)
    
    for path in s_path_list:
        if event in path:
            return path
    
    raise Exception('Cannot find the file for this event')


def find_multiple_path(event, dir):
    result = []
    path_list = glob.glob(dir)
    s_path_list = sorted(path_list)
    
    for path in s_path_list:
        if event in path:
            result.append(path)
    
    return result


def extract_files(event):
    """
    Wrapper function for finding the files.
    """
    alt_PE_runs = ['GW170608', 'GW190707', 'GW190720', 'GW190728', 'GW190814', 'GW190924']
    
    if event in alt_PE_runs:
        result_dir = os.path.dirname(os.path.dirname(__file__))+"/other_PE_run/*.json"
        data_dir = os.path.dirname(os.path.dirname(__file__))+"/other_PE_run/data/*"
        result_file_path = find_path(event, dir=result_dir)
        data_file_path = find_path(event, dir=data_dir)
    else:
        result_dir = os.path.dirname(os.path.dirname(__file__))+"/GWOSC_PE_runs/posteriors/*.h5"
        data_dir = None
        result_file_path = find_path(event, dir=result_dir)
        data_file_path = None
        
    return result_file_path, data_file_path
