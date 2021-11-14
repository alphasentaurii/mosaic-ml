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
from tqdm import tqdm
from progressbar import ProgressBar
from make_images import generate_total_images, generate_filter_images

SVM_QUALITY_TESTING="on"


def pick_random_exposures(dataset):
    hapfiles = glob.glob(f"{dataset}/*.fits")
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
    drz_file = glob.glob(f"{dataset}/*.out")[0]
    drizzle_dct = {}
    with open(drz_file, 'r') as f:
        input_strings = f.readlines()
        for s in input_strings:
            tokens = s.split(',')
            drizzle_file = f"{dataset}/{tokens[0]}"
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
            new = f"{name}/{filename}"
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
    

def artificial_misalignment(dataset, selector, outputs):
    dname = dataset.split("/")[-1]
    name = f"{outputs}/{dname}_{selector}"
    shutil.copytree(dataset, name)
    if selector == "rex":
        selected_files = pick_random_exposures(name)
    elif selector == "rfi":
        selected_files = pick_random_filter(name)
    run_header_corruption(selected_files)


def multiple_permutations(dataset, exp, mode, outputs, level="any"):
    drizzle_dct = find_filter_files(dataset)
    filters = list(drizzle_dct.keys())
    separator = "---"*5
    bar = ProgressBar().start()

    for x, f in zip(bar(range(len(filters))),filters):
        dname = dataset.split("/")[-1]
        name = f"{outputs}/{dname}_{f.lower()}_{exp}_{mode}"
        if not os.path.exists(name):
            shutil.copytree(dataset, name)
        drizzle_mod = modify_paths(drizzle_dct, name)
        out = sys.stdout
        err = 0
        with open(f"{name}/corruption.txt", 'w') as logfile:
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
            with open(f"{name}/warning.txt", 'w') as warning:
                sys.stdout = warning
                print("WARNING: only 1 exposure but you requested a subset")
                sys.stdout = out
        bar.update(x)
    bar.finish()


# experimental (Run SVM via command line instead)
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


# experimental (Run SVM via command line instead)
def generate_images(dataset, filters=False):
    input_path = os.getcwd()
    generate_total_images(input_path, datasets=[dataset], output_img='./img/total/1')
    if filters is True:
        generate_filter_images(input_path, dataset=dataset, outpath='./img/filter', figsize=(24,24), crpt=0)


def all_permutations(dataset, outputs):
    multiple_permutations(dataset, "all", "stat", outputs)
    multiple_permutations(dataset, "all", "stoc", outputs)
    multiple_permutations(dataset, "sub", "stat", outputs)
    multiple_permutations(dataset, "sub", "stoc", outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MosaicML", usage="python corrupt.py j8ep07 mfi -e=sub -m=stat")
    parser.add_argument("srcpath", type=str, help="parent directory of single visit datasets")
    parser.add_argument("outputs", type=str, help="path for saving corrupted hap file copies")
    parser.add_argument("selector", type=str, choices=["rex", "rfi", "mfi", "multi"], help="`rex`: randomly select subset of exposures from any filter; `rfi`: select all exposures from randomly selected filter; `mfi`: exposures of one filter, repeated for every filter in dataset. 'multi' creates sub- and all- MFI permutations")
    parser.add_argument("-p", "--pattern", type=str, default="*", help="glob search pattern - default is wildcard *")
    parser.add_argument("-e", "--exposures", type=str, choices=["all", "sub"], default="all", help="all or subset of exposures")
    parser.add_argument("-m", "--mode", type=str, choices=["stat", "stoc"], default="stoc", help="apply consistent (static) or randomly varying (stochastic) corruptions to each exposure")
    parser.add_argument("-l", "--level", type=str, choices=["major", "standard", "minor", "any"], default="any", help="lambda relative error level")
    # get user-defined args and/or set defaults
    args = parser.parse_args()
    srcpath, outputs, selector =  args.srcpath, args.outputs, args.selector
    pattern, expo, mode = args.pattern, args.exposures, args.mode
    datasets = glob.glob(f"{srcpath}/{pattern}")
    if len(datasets) < 1:
        print("No datasets found matching the search pattern.")
        sys.exit(1)
    else:
        for dataset in tqdm(datasets):
            if selector == "multi":
                all_permutations(dataset, outputs)
            elif selector == "mfi":
                multiple_permutations(dataset, expo, mode, outputs, level=args.level)
            elif selector in ["rex", "rfi"]:
                artificial_misalignment(dataset, selector, outputs)
