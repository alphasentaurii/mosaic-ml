"""
CRVAL1 and CRVAL2 give the center coordinate as right ascension and declination or longitude and latitude in decimal degrees.

CRPIX1 and CRPIX2 are the pixel coordinates of the reference point to which the projection and the rotation refer.

The default corruption method is on CRVAL1 and/or CRVAL2 only. Corruption can also be performed on CRPIX values; this would allow a machine learning algorithm to be trained on an alternative (albeit more rare) cause of single visit mosaic misalignment.

"""
import os
import sys
import argparse
import shutil
import glob
import numpy as np
from astropy.io import fits
import subprocess


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
    n_corruptions = np.random.randint(2, n_files)
    if n_files < n_corruptions:
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
    shutil.copytree(dataset, dataset+"_corr")
    if selector == "rex":
        dataset = dataset+"_rex"
        selected_files = pick_random_exposures(dataset)
    elif selector == "rfi":
        dataset = dataset+"_rfi"
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
        with open(f"./{name}/corruption.txt", 'w') as logfile:
            sys.stdout = logfile
            print(separator)
            print("\nFILTER: ", f)
            if exp == "all":
                selected_files = drizzle_mod[f]
                print("\nALL FILES: ", selected_files)
            else:
                filter_files = drizzle_mod[f]
                selected_files = pick_random_subset(filter_files)
            run_header_corruption(selected_files, level=level, mode=mode)
            sys.stdout = out


def run_svm(dataset):
    mutations = glob.glob(f"{dataset}_*")
    cwd = os.getcwd()
    for m in mutations:
        os.chdir(m)
        drz_file = glob.glob(f"*.out")[0]
        cmd = ["runsinglehap", drz_file]
        err = subprocess.call(cmd)
        if err:
            print(f"SVM failed to run for {m}")
        os.chdir(cwd)


def get_files_for_image_gen(dataset):
    corr_folder = f"./{dataset}_corrs"
    os.mkdir(corr_folder, exist_ok=True)
    mutations = glob.glob(f"{dataset}_*")
    for m in mutations:
        fits_file = glob.glob(f"{m}/hst_*_total_{dataset}_dr?.fits")[0]
        p_cat = glob.glob(f"{m}/hst_*_total_{dataset}_point-cat.ecsv")[0]
        s_cat = glob.glob(f"{m}/hst_*_total_{dataset}_segment-cat.ecsv")[0]
        g_cat = glob.glob(f"{m}/hst_*_GAIAeDR3_ref_cat.ecsv")[0]
        F, G = f"{corr_folder}/{m}/fits", f"{corr_folder}/{m}/cat/gaia"
        P, S = f"{corr_folder}/{m}/cat/point", f"{corr_folder}/{m}/cat/segment"
        dirs = [F, G, P, S]
        for d in dirs:
            os.makedirs(d)
        shutil.copy(fits_file, f"{F}/"+os.path.basename(fits_file))
        shutil.copy(g_cat, f"{G}/"+os.path.basename(g_cat))
        shutil.copy(p_cat, f"{P}/"+os.path.basename(p_cat))
        shutil.copy(s_cat, f"{S}/"+os.path.basename(s_cat))


def generate_images(dataset):
    corr_folder = f"./{dataset}_corrs"
    mutations = glob.glob(f"{corr_folder}/{dataset}_*")
    cwd = os.getcwd()
    for m in mutations:
        os.chdir(m)
        cmd = ["python", "make_images.py", "fits", "-i", "total", "-g", "1"]
        err = subprocess.call(cmd)
        if err:
            print(f"Image Generator error for {m}")
        os.chdir(cwd)


#TODO: drizzlepac h5 file creator from json files
def make_h5_file():
    mutations = glob.glob(f"{dataset}_*")
    for m in mutations:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to SVM dataset directory")
    parser.add_argument("selector", type=str, choices=["rex", "rfi", "multi"], help="rex: randomly select subset of exposures from any filter; rfi: select all exposures from randomly selected filter; multi: all exposures of a single filter, repeated for each filter (creates multiple corruption permutations")
    parser.add_argument("-e", "--exposures", type=str, choices=["all", "sub"], help="all or subset of exposures")
    parser.add_argument("-m", "--mode", type=str, choices=["stat", "stoc"], help="apply consistent (static) or randomly varying (stochastic) corruptions to each exposure")
    parser.add_argument("-l", "--level", type=str, choices=["major", "standard", "minor", "any"], help="lambda relative error level")
    # get user-defined args and/or set defaults
    args = parser.parse_args()
    dataset, selector = args.dataset, args.selector
    exp, mode = args.exposures, args.mode
    level = args.level
    if exp is None:
        exp = "all"
    if mode is None:
        mode = "stoc"
    if level is None:
        level = "any"
    if selector == "multi":
        multiple_permutations(dataset, exp, level, mode)
    else:
        artificial_misalignment(dataset, selector)
