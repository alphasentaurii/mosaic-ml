import pandas as pd
import argparse
import os
import sys
from astropy.io import fits
from astroquery.mast import Observations
import numpy as np
from sklearn.preprocessing import LabelEncoder
from stsci.tools import logutil
from progressbar import ProgressBar
import harvest as djh
import json

__taskname__ = 'mosaic_ml_data_import'

MSG_DATEFMT = '%Y%j%H%M%S'
SPLUNK_MSG_FORMAT = '%(asctime)s %(levelname)s src=%(name)s- %(message)s'
log = logutil.create_logger(__name__, level=logutil.logging.NOTSET, stream=sys.stdout,
                            format=SPLUNK_MSG_FORMAT, datefmt=MSG_DATEFMT)

log_dict = {"critical": logutil.logging.CRITICAL,
            "error": logutil.logging.ERROR,
            "warning": logutil.logging.WARNING,
            "info": logutil.logging.INFO,
            "debug": logutil.logging.DEBUG}

def make_h5_file(data_path, patterns=['*_total*_svm_*.json'], hdf5_file='ml_train', crpt=0):
    print("*** Starting JSON Harvest ***")
    djh.json_harvester(
        json_search_path=data_path,
        json_patterns=patterns,
        output_filename_basename=hdf5_file,
        crpt=crpt
        )
    if not hdf5_file.endswith(".h5"):
        hdf5_file += '.h5'
    return hdf5_file


def load_h5_file(h5_file):
    if not h5_file.endswith(".h5"):
        h5_file += '.h5'
    if os.path.exists(h5_file):
        with pd.HDFStore(h5_file) as store:
            data = store['mydata']
            print(f"Dataframe created: {data.shape}")
    else:
        errmsg = "HDF5 file {} not found!".format(h5_file)
        log.error(errmsg)
        raise Exception(errmsg)
    return data


def split_index(df):
    idx_dct = {}
    for idx, _ in df.iterrows():
        n = str(idx)
        items = n.split('_')
        idx_dct[n] = {}
        idx_dct[n]['detector'] = items[4]
        if len(items) > 7:
            idx_dct[n]['dataset'] = '_'.join(items[6:])
            # idx_dct[n]['dataset'] = items[6][:6] + '_' + items[7]
        else:
            idx_dct[n]['dataset'] = items[6]
            #idx_dct[n]['dataset'] = items[6][:6]
    df_index = pd.DataFrame.from_dict(idx_dct, orient='index')
    df = df_index.join(df, how='left')
    return df


def rename_cols(df, training_cols, splitter='.'):
    log.info("Renaming columns")
    cols = [col.split(splitter)[-1].lower() for col in df.columns]
    hc = dict(zip(df.columns, cols))
    df.rename(hc, axis='columns', inplace=True)
    extract = [c for c in training_cols if c in df.columns]
    df = df[extract]
    log.info("New column names: ", df.columns)
    return df


def extract_columns(df):
    print("*** Extracting FITS header prefix columns ***")
    prefix = ['header', 'gen_info', 'number_of_sources', 'Number_of_GAIA_sources.']
    extract_cols = []
    for c in prefix:
        extract_cols += [col for col in df if c in col]
    train_cols = [
        'targname', 
        'ra_targ', 
        'dec_targ', 
        'numexp', 
        'imgname', 
        'point', 
        'segment', 
        'number_of_gaia_sources'
        ]
    df = rename_cols(df[extract_cols], train_cols)
    df.dropna(axis=0, inplace=True)
    df = split_index(df)
    return df


def extract_alignment_data(df, data_path):
    print("*** Extracting alignment data ***")
    drz_paths = {}
    for idx, row in df.iterrows():
        drz_paths[idx] = ''
        #dname =  row['dataset']
        dname = '_'.join(row['dataset'].split('-'))
        drz = row['imgname']
        path = os.path.join(data_path, dname, drz)
        drz_paths[idx] = path
    align_dct = {}
    keywords = ['rms_ra', 'rms_dec', 'nmatches', 'wcstype']
    for key, path in drz_paths.items():
        align_dct[key] = {}
        scihdr = fits.getheader(path, ext=1)
        for k in keywords:
            if k in scihdr:
                if k == 'wcstype':
                    wcs = ' '.join(scihdr[k].split(' ')[1:3])
                    align_dct[key][k] = wcs
                else:
                    align_dct[key][k] = scihdr[k]
            else:
                align_dct[key][k] = 0
    align_data = pd.DataFrame.from_dict(align_dct, orient='index')
    log.info(f"Alignment data added:\n {align_data.info()}")
    df = df.join(align_data, how='left')
    return df


def find_category(df):
    print("*** Assigning target name categories ***")
    target_categories = {}
    targets = df['targname'].unique()
    log.info(f"Unique Target Names: {len(targets)}")
    bar = ProgressBar().start()
    for x, targ in zip(bar(range(len(targets))),targets):
        if targ != 'ANY':
            obs = Observations.query_criteria(target_name=targ)
            cat = obs[np.where(obs['target_classification'])]['target_classification']
            if len(cat) > 0:
                target_categories[targ] = cat[0]
            else:
                target_categories[targ] = 'None'
        bar.update(x)
    bar.finish()

    other_cat = {}
    targ_any = df.loc[df['targname'] == 'ANY'][['ra_targ', 'dec_targ']]
    log.info(f"Other targets (ANY): {len(targ_any)}")
    if len(targ_any) > 0:
        bar = ProgressBar().start()
        for x, (idx, row) in zip(bar(range(len(targ_any))), targ_any.iterrows()):
            other_cat[idx] = {}
            propid = str(idx).split('_')[1]
            ra, dec = row['ra_targ'], row['dec_targ']
            obs = Observations.query_criteria(proposal_id=propid, s_ra=ra, s_dec=dec)
            cat = obs[np.where(obs['target_classification'])]['target_classification']
            if len(cat) > 0:
                other_cat[idx] = cat[0]
            else:
                other_cat[idx] = 'None'
            bar.update(x)
        bar.finish()

    categories = {}
    for k, v in target_categories.items():
        idx = df.loc[df['targname'] == k].index
        for i in idx:
            categories[i] = v
    categories.update(other_cat)
    df_cat = pd.DataFrame.from_dict(categories, orient='index', columns={'category'})
    log.info(f"Target Categories Assigned \n {df_cat['category'].value_counts()}")
    df = df.join(df_cat, how='left')
    return df


def encode_categories(df, sep=';'):
    print("*** Encoding Category Names ***")
    CAT = {}
    category_keys = {
            'CALIBRATION': 'C',
            'SOLAR SYSTEM' : 'SS',
            'ISM': 'I',
            'EXT-MEDIUM': 'I',
            'UNIDENTIFIED': 'U',
            'STELLAR CLUSTER': 'SC',
            'EXT-CLUSTER': 'SC',
            'STAR': 'S',
            'EXT-STAR': 'S',
            'CLUSTER OF GALAXIES': 'GC',
            'GALAXY': 'G'            
        }
    for idx, cat in df.category.items():
        c = cat.split(sep)[0]
        if c in category_keys:
            CAT[idx] = category_keys[c]
    df_cat = pd.DataFrame.from_dict(CAT, orient='index', columns={'cat'})
    log.info("Category encoding complete.\n", df_cat['cat'].value_counts())
    df = df.join(df_cat, how='left')
    return df


def encode_features(df, encodings):
    for col, name in encodings.items():
        encoder = LabelEncoder().fit(df[col])
        df[name] = encoder.transform(df[col])
        print(df[name].value_counts())
    return df


def find_subsamples(df, output_file):
    if 'label' not in df.columns:
        return
    df = df.loc[df['label'] == 0]
    subsamples = {}
    categories = list(df['cat'].unique())
    detectors = list(df['det'].unique())
    for d in detectors:
        det = df.loc[df['det'] == d]
        for c in categories:
            cat = det.loc[det['cat'] == c]
            if len(cat) > 0:
                idx = np.random.randint(0, len(cat))
                subsamples[f'c{c}_d{d}'] = cat.index[idx]
            else:
                continue
    index = list(subsamples.values())
    datasets = []
    for i in index:
        dataset = i.split('_')[-2]
        datasets.append(dataset)
    output_path = os.path.dirname(output_file)
    with open(f'{output_path}/subsamples.txt', 'w') as j:
        json.dump(subsamples, j)


def build_raw(data, data_path, outpath, outfile):
    print("*** Extracting Raw Data ***")
    df = extract_columns(data)
    df = extract_alignment_data(df, data_path)
    df = find_category(df)
    # save raw_data before preprocessing
    df['index'] = df.index
    df.to_csv(f'{outpath}/raw_{outfile}', index=False)
    df.set_index('index', inplace=True)
    return df

def encode_data(df, crpt):
    df = encode_categories(df)
    encodings = {'wcstype': 'wcs', 'cat': 'cat', 'detector': 'det'}
    df = encode_features(df, encodings)
    drops = ['detector', 'category', 'wcstype', 'targname','ra_targ', 'dec_targ', 'imgname', 'dataset']
    df = df.drop(drops, axis=1)
    df.rename({'number_of_gaia_sources':'gaia'}, axis=1, inplace=True)
    if crpt:
        labels = []
        for _ in range(len(df)):
            labels.append(1)
        df['label'] = pd.Series(labels).values
    else:
        find_subsamples(df, output_file)
    return df


def set_columns(df, outpath):
    column_order = ['numexp', 'rms_ra', 'rms_dec', 'nmatches', 'point', 'segment', 'gaia', 'det', 'wcs', 'cat']
    if 'label' in df.columns:
        column_order.append('label')
        pos = list(df.loc[df['label'] == 1].index.values)
        if len(pos) > 0:
            with open(f'{outpath}/pos.txt', 'w') as f:
                for i in pos:
                    f.writelines(f"{i}\n")
    df = df[column_order]
    return df


def main(hdf5_file, output_file, data_path, make, crpt, log_level):
    log.setLevel(log_level)
    outpath = os.path.dirname(output_file)
    outfile = os.path.basename(output_file)
    os.makedirs(outpath, exist_ok=True)
    if make:
        hdf5_file = os.path.join(outpath, hdf5_file)
        hdf5_file = make_h5_file(data_path, hdf5_file=hdf5_file, crpt=crpt)
    data = load_h5_file(hdf5_file)
    df = build_raw(data, data_path, outpath, outfile)
    df = encode_data(df, crpt)
    df = set_columns(df, outpath)
    df['index'] = df.index
    df.to_csv(output_file, index=False)
    print(f"Dataframe saved to CSV: {output_file}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Mosaic ML Data Import", usage="python make_dataset.py ml_train -d=singlevisits -o=svm.csv")
    parser.add_argument("hdf5", type=str, default='ml_train_dataframe', help="hdf5 filepath")
    parser.add_argument("-d", "--datapath", type=str, default="./data/singlevisits", help="svm datasets directory path")
    parser.add_argument("-o","--output", type=str, default="./data/svm_data.csv", help="csv output filepath")
    parser.add_argument("-m","--make", type=str, default=1, help="make hdf5 file from json files")
    parser.add_argument("-l", "--loglevel", type=str, default="info", help="set log level")
    parser.add_argument("-c", "--crpt", type=int, default=0, choices=[0,1], help="set to 1 for corruption data")
    args = parser.parse_args()
    hdf5_file = args.hdf5
    data_path = args.datapath
    output_file = args.output
    log_level=log_dict[args.loglevel]
    main(hdf5_file, output_file, data_path, args.make, args.crpt, log_level)
    

    
