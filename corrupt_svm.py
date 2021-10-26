"""
CRVAL1 and CRVAL2 give the center coordinate as right ascension and declination or longitude and latitude in decimal degrees.

CRPIX1 and CRPIX2 are the pixel coordinates of the reference point to which the projection and the rotation refer.

The default corruption method is on CRVAL1 and/or CRVAL2 only. Corruption can also be performed on CRPIX values; this would allow a machine learning algorithm to be trained on an alternative (albeit more rare) cause of single visit mosaic misalignment.

"""
import os
import sys
import argparse
import shutil
import subprocess
import glob
import numpy as np
from astropy.io import fits
from drizzlepac.haputils import diagnostic_json_harvester as djh

from make_images import generate_total_images, draw_filter_images

SVM_QUALITY_TESTING="on"

def pick_random_exposures(dataset):
    hapdir = os.path.join('.', dataset)
    hapfiles = glob.glob(f"{hapdir}/*.fits")
    n_corruptions = np.random.randint(2, 4)
    if len(hapfiles) < n_corruptions:
        n_corruptions = max(len(hapfiles), 2)
    print(f"Selecting {n_corruptions} out of {len(hapfiles)} files")
    np.random.shuffle(hapfiles)
    cor_idx = []
    for _ in list(range(n_corruptions)):
        cor_idx.append(np.random.randint(len(hapfiles)))
    print(f"Shuffled random index: {cor_idx}")
    selected_files = []
    for i, j in enumerate(hapfiles):
        if i in cor_idx:
            selected_files.append(j)
    print(f"Files selected for corruption: ", selected_files)
    return selected_files


def pick_random_filter(dataset):
    drizzle_dct = find_filter_files(dataset)
    i = np.random.randint(0, len(drizzle_dct))
    f = list(drizzle_dct.keys())[i]
    print(f"\nRANDOM FILTER SELECTED: {f}")
    drz_files = drizzle_dct[f]
    print(f"\nFILES SELECTED: {drz_files}")
    return drz_files


def find_filter_files(dataset):
    hapdir = os.path.join('.', dataset)
    drz_file = glob.glob(f"{hapdir}/*.out")[0]
    drizzle_dct = {}
    with open(drz_file, 'r') as f:
        input_strings = f.readlines()
        for s in input_strings:
            tokens = s.split(',')
            drizzle_file = f"{hapdir}/{tokens[0]}"
            fltr = tokens[-3].replace(';', '_')
            if fltr not in drizzle_dct:
                drizzle_dct[fltr] = [drizzle_file]
            else:
                drizzle_dct[fltr].append(drizzle_file)
    return drizzle_dct


def modify_paths(drizzle_dct, name):
    drizzle_mod = {}
    for flt, paths in drizzle_dct.items():
        drizzle_mod[flt] = []
        for p in paths:
            filename = p.split('/')[-1]
            new = f"./{name}/{filename}"
            drizzle_mod[flt].append(new)
    return drizzle_mod


def pick_random_subset(filter_files):
    n_files = len(filter_files)
    if n_files > 1:
        n_corruptions = np.random.randint(1, n_files)
    else:
        print(f"WARNING - {n_files} exposure for this filter.")
        n_corruptions = n_files
    print(f"\nSelecting {n_corruptions} out of {n_files} files")
    np.random.shuffle(filter_files)
    cor_idx = []
    for _ in list(range(n_corruptions)):
        cor_idx.append(np.random.randint(n_files))
    print(f"\nShuffled random index: {cor_idx}")
    selected_files = []
    for i, j in enumerate(filter_files):
        if i in cor_idx:
            selected_files.append(j)
    print(f"\nFiles selected for corruption: ", selected_files)
    return selected_files


# def random_integer_shift(x):
#     return np.round(x + np.random.randint(-2, 2))


def set_lambda_threshold(level):
    if level == "major":
        return np.random.uniform(10, 20)
    elif level == "standard":
        return np.random.uniform(0.5, 10)
    elif level == "minor":
        return np.random.uniform(0, 1)
    else:
        return np.random.uniform(0, 20)


def static_augment(level):
    lamb = set_lambda_threshold(level)
    delta = (lamb*0.04)/3600
    p, r = np.random.uniform(0,1), np.random.uniform(0,1)
    print("PIXEL offset: ", lamb)
    if r < p:
        print("DEGREE offset: +", delta)
        return delta
    else:
        print("DEGREE offset: -", delta)
        return -delta


def static_corruption(fits_file, delta):
    print("\nApplying static augment: ", fits_file)
    with fits.open(fits_file, 'update') as hdu: 
        wcs_valid = {
            'CRVAL1': hdu[1].header['CRVAL1'],
            'CRVAL2': hdu[1].header['CRVAL2']
            }
        wcs_corrupt = wcs_valid.copy()
        wcs_corrupt['CRVAL1'] += delta
        wcs_corrupt['CRVAL2'] += delta
        hdu[1].header['CRVAL1'] = wcs_corrupt['CRVAL1']
        hdu[1].header['CRVAL2'] = wcs_corrupt['CRVAL2']
    return wcs_valid, wcs_corrupt


def stochastic_augment(x, level):
    lamb = set_lambda_threshold(level)
    delta = (lamb*0.04)/3600
    p, r = np.random.uniform(0,1), np.random.uniform(0,1)
    print("\nCRVAL: ", x)
    print("PIXEL offset: ", lamb)
    if r < p:
        print("DEGREE offset: +", delta)
        return x + delta
    else:
        print("DEGREE offset: -", delta)
        return x - delta


def stochastic_corruption(fits_file, level):
    print("\nApplying stochastic augment: ", fits_file)
    with fits.open(fits_file, 'update') as hdu: 
        wcs_valid = {
            'CRVAL1': hdu[1].header['CRVAL1'],
            'CRVAL2': hdu[1].header['CRVAL2']
            }
        wcs_corrupt = wcs_valid.copy()
        wcs_corrupt['CRVAL1'] = stochastic_augment(wcs_corrupt['CRVAL1'], level)
        wcs_corrupt['CRVAL2'] = stochastic_augment(wcs_corrupt['CRVAL2'], level)
        hdu[1].header['CRVAL1'] = wcs_corrupt['CRVAL1']
        hdu[1].header['CRVAL2'] = wcs_corrupt['CRVAL2']
    return wcs_valid, wcs_corrupt


def print_corruptions(wcs_valid, wcs_corrupt):
    separator = "---!@#$%^&*()_+---"*3
    print("\nCRVAL1-old: ", wcs_valid['CRVAL1'])
    print("CRVAL1-new: ", wcs_corrupt['CRVAL1'])
    print("\nCRVAL2-old: ", wcs_valid['CRVAL2'])
    print("CRVAL2-new: ", wcs_corrupt['CRVAL2'])
    print(f"\n{separator}")


def run_header_corruption(selected_files, level="any", mode="stoc"):
    if mode == "stat":
        print("\nStarting static corruption\n")
        delta = static_augment(level)
        for fits_file in selected_files:
            wcs_valid, wcs_corrupt = static_corruption(fits_file, delta)
            print_corruptions(wcs_valid, wcs_corrupt)
    else:
        print("\nStarting stochastic corruption\n")
        for fits_file in selected_files:
            wcs_valid, wcs_corrupt = stochastic_corruption(fits_file, level)
            print_corruptions(wcs_valid, wcs_corrupt)
    

def artificial_misalignment(dataset, selector):
    shutil.copytree(dataset, f"{dataset}_{selector}")
    dataset = f"{dataset}_{selector}"
    if selector == "rex":
        selected_files = pick_random_exposures(dataset)
    elif selector == "rfi":
        selected_files = pick_random_filter(dataset)
    run_header_corruption(selected_files)

 
def multiple_permutations(dataset, exp, level, mode):
    drizzle_dct = find_filter_files(dataset)
    filters = list(drizzle_dct.keys())
    separator = "---"*5
    for f in filters:
        name = f"{dataset}_{f.lower()}_{exp}_{mode}"
        shutil.copytree(dataset, name)
        drizzle_mod = modify_paths(drizzle_dct, name)
        out = sys.stdout
        err = 0
        with open(f"./{name}/corruption.txt", 'w') as logfile:
            sys.stdout = logfile
            print(separator)
            print("\nFILTER: ", f)
            if exp == "all":
                selected_files = drizzle_mod[f]
                print("\nALL FILES: ", selected_files)
            else:
                filter_files = drizzle_mod[f]
                if len(filter_files) == 1:
                    err += 1
                selected_files = pick_random_subset(filter_files)
            run_header_corruption(selected_files, level=level, mode=mode)
            sys.stdout = out
        if err == 1:
            with open(f"./{name}/warning.txt", 'w') as warning:
                sys.stdout = warning
                print("WARNING: only 1 exposure but you requested a subset")
                sys.stdout = out



def run_svm(dataset):
    os.environ.get('SVM_QUALITY_TESTING', "on")
    mutations = glob.glob(f"{dataset}_*")
    cwd = os.getcwd()
    for m in mutations:
        os.chdir(m)
        warning = f"./warning.txt"
        if os.path.exists(warning):
            print(f"Skipping {m} - see warning file")
        else:
            drz_file = glob.glob(f"*.out")[0]
            cmd = ["runsinglehap", drz_file]
            err = subprocess.call(cmd)
            if err:
                print(f"SVM failed to run for {m}")
        os.chdir(cwd)


def rename_copy_files(dataset, m, f_dir, g_dir, p_dir, s_dir):
    mm = '_'.join(m.split('_')[1:])
    f1 = glob.glob(f"{m}/hst_*_total_{dataset}_dr?.fits")
    p1 = glob.glob(f"{m}/*total*point-cat.ecsv")
    s1 = glob.glob(f"{m}/*total*segment-cat.ecsv")
    g1 = glob.glob(f"{m}/*GAIA*.ecsv")
    file_pairs = dict(f_dir=f1, g_dir=g1, p_dir=p1, s_dir=s1)
    missing = []
    for k, v in file_pairs.items():
        if len(v) > 0:
            v = v[0]
            v2 = f"{k}/hst_{mm}_{'_'.join(os.path.basename(v).split('_')[4:])}"
            shutil.copy(v, v2)
        else:
            missing.append((m, k))
    print(missing)
    return file_pairs


def get_total_image_gen_files(dataset):
    #TODO: check for existing paths first, handle exceptions
    mutations = glob.glob(f"{dataset}_*")
    c_dir = f"./total_corrs"
    f_dir, g_dir = f"{c_dir}/fits", f"{c_dir}/cat/gaia"
    p_dir, s_dir, = f"{c_dir}/cat/point", f"{c_dir}/cat/segment"
    for d in [c_dir, f_dir, g_dir, p_dir, s_dir]:
        os.makedirs(d, exist_ok=True)
    
    for m in mutations:
        # copy and rename relevant fits and catalog files
        mm = '_'.join(m.split('_')[1:])
        try:
            f1 = glob.glob(f"{m}/hst_*_total_{dataset}_dr?.fits")[0]
            p1 = glob.glob(f"{m}/*total*point-cat.ecsv")[0]
            s1 = glob.glob(f"{m}/*total*segment-cat.ecsv")[0]
            g1 = glob.glob(f"{m}/*GAIA*.ecsv")[0]
            f2 = f"{f_dir}/hst_{mm}_{'_'.join(os.path.basename(f1).split('_')[4:])}"
            p2 = f"{p_dir}/hst_{mm}_{'_'.join(os.path.basename(p1).split('_')[4:])}"
            s2 = f"{s_dir}/hst_{mm}_{'_'.join(os.path.basename(s1).split('_')[4:])}"
            g2 = f"{g_dir}/hst_{mm}_{'_'.join(os.path.basename(g1).split('_')[4:])}"
            files = [(f1, f2), (g1, g2), (p1, p2), (s1, s2)]
            for (a, b) in files:
                if os.path.exists(a):
                    shutil.copy(a, b)
                else:
                    print(f"Couldn't find source file {a}")
        except IndexError as e:
            print(m, e)
    # for root, _, f in os.walk(c_dir):
    #     indent = "    " * root.count(os.sep)
    #     print("{}{}/".format(indent, os.path.basename(root)))
    #     for filename in f:
    #         print("{}{}".format(indent + "    ", filename))
    return c_dir

def get_filter_image_gen_files(dataset):
    c_dir = f"./{dataset}_filters"
    fits_dir = f"{c_dir}/fits"
    os.makedirs(fits_dir, exist_ok=True)
    mutations = glob.glob(f"{dataset}_*")
    for m in mutations:
        mm = '_'.join(m.split('_')[1:])
        splits = mm.split('_')
        if len(splits) > 3:
            flt = splits[:2]
        else:
            flt = splits[0]
        flt1 = glob.glob(f"{m}/hst_*_{flt}_{dataset}_dr?.fits")[0]
        flt2 = f"{fits_dir}/hst_{mm}_{'_'.join(os.path.basename(flt1).split('_')[4:])}"
        shutil.copy(flt1, flt2)
    return c_dir


def generate_images(dataset, filters=False):
    if filters is True:
        c_dir = get_filter_image_gen_files(dataset)
        filter_path = f"{c_dir}/fits"
        filter_files = os.listdir(filter_path)
        draw_filter_images(filter_files, filter_path, figsize=(24,24))
    else:
        c_dir = get_total_image_gen_files(dataset, filters=filters)
        generate_total_images(f"{c_dir}/fits", generator=1)

    # mutations = glob.glob(f"{c_dir}/{dataset}_*")
    # for m in mutations:
    #     cmd = ["python", "make_images.py", f"{m}/fits", "-i", "total", "-g", "1"]
    #     err = subprocess.call(cmd)
    #     if err:
    #         print(f"Image Generator error for {m}")


def make_h5_file(svm_data):
    os.chdir(svm_data)
    djh.json_harvester(
        json_search_path='./', 
        json_patterns=['*_svm_*.json'], 
        output_filename_basename='ml_train_dataframe'
        )
# def make_h5_file(dataset):
#     """
#     creates h5 file from SVM-generated json QA files
#     """
#     for visit in os.listdir(dataset):
#     mutations = glob.glob(f"{dataset}_*")
#     for m in mutations:
#         pass
# from drizzlepac.haputils import diagnostic_json_harvester as djh
#     djh.json_harvester(
#         json_search_path='./', 
#         json_patterns=['*_svm_*.fits'], 
#         output_filename_basename='ml_train_dataframe'
#         )
# replace the *_svm_*.fits  with the patterns for the JSON filenames that you want included in the HDF5 file.  
# This should only be related JSON files, such as only JSON files generated during SVM processing with svm in the filename or only JSON files generated during MVM processing with mvm in the name.  
# It will also pull JSON files from all sub-directories if that is what you specify through the use of glob in order to allow you to call it once in the parent directory for multiple sub-directories worth of results to get just one combined HDF file. 

def run_multiple(dataset_directory, runsvm=1, imagegen=1):
    visits = os.listdir(dataset_directory)
    for visit in visits:
        multiple_permutations(visit, "all", "any", "stat")
        multiple_permutations(visit, "all", "any", "stoc")
        multiple_permutations(visit, "sub", "any", "stat")
        multiple_permutations(visit, "sub", "any", "stoc")
        if runsvm == 1:
            run_svm(visit)
        if imagegen == 1:
            generate_images(visit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Corrupt SVM", usage="python corrupt_svm.py j8ep07 multi -e=sub -m=stoc")
    parser.add_argument("dataset", type=str, help="path to SVM dataset directory (single visit or collection of visits if using `selector=multi`")
    parser.add_argument("selector", type=str, choices=["rex", "rfi", "mfi", "multi"], help="`rex`: randomly select subset of exposures from any filter; `rfi`: select all exposures from randomly selected filter; `mfi`: exposures of one filter, repeated for every filter in dataset; `multi`: creates sub- and all- MFI permutations for group of datasets")
    parser.add_argument("-e", "--exposures", type=str, choices=["all", "sub"], default="all", help="all or subset of exposures")
    parser.add_argument("-m", "--mode", type=str, choices=["stat", "stoc"], default="stoc", help="apply consistent (static) or randomly varying (stochastic) corruptions to each exposure")
    parser.add_argument("-l", "--level", type=str, choices=["major", "standard", "minor", "any"], default="any", help="lambda relative error level")
    parser.add_argument("-r", "--runsvm", type=int, choices=[0,1], default=0, help="Run SVM on corrupted dataset(s)")
    parser.add_argument("-i", "--imagegen", type=int, choices=[0,1], default=0, help="generate images (runsvm must also be set to 1)")
    # get user-defined args and/or set defaults
    args = parser.parse_args()
    dataset, selector = args.dataset, args.selector
    exp, mode, level = args.exposures, args.mode, args.level
    # all-in-one shot: 'dataset' points to directory of multiple visits and script generates all permutations for each one
    if selector == "multi":
        run_multiple(dataset, runsvm=args.runsvm, imagegen=args.imagegen)
    elif selector == "mfi":
        multiple_permutations(dataset, exp, level, mode)
    elif selector in ["rex", "rfi"]:
        artificial_misalignment(dataset, selector)
    else:
        print("Selector must be one of: rex, rfi, mfi, multi")
