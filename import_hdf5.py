""" the SVM "harvester file" which contains a Pandas dataframe stored in an 
    HDF5 file.  The harvester file is the collection of JSON output generated 
    by the SVM quality analysis tests.
"""
# import logging
# from stsci.tools import logutil

# MSG_DATEFMT = '%Y%j%H%M%S'
# SPLUNK_MSG_FORMAT = '%(asctime)s %(levelname)s src=%(name)s- %(message)s'
# log = logutil.create_logger(__name__, level=logutil.logging.NOTSET, stream=sys.stdout,
#                             format=SPLUNK_MSG_FORMAT, datefmt=MSG_DATEFMT)
import pandas as pd
import sys
import os
import shutil


DETECTOR_LEGEND = {'UVIS': 'magenta', 'IR': 'red', 'WFC': 'blue',
                'SBC': 'yellow', 'HRC': 'black'}

def svm_harvester(h5_file, save_csv=None):
    h5_dir = './data/h5files'
    os.makedirs(h5_dir, exist_ok=True)
    filename = os.path.basename(h5_file)
    svm_file = os.path.join(h5_dir, filename)
    shutil.copy(h5_file, svm_file)
    
    hdf5 = pd.HDFStore(svm_file, mode='r')
    print(hdf5)
    key0 = hdf5.keys()[0] # '/mydata'
    print(key0)
    df = hdf5.get(key0)
    print(f"Dataframe created: {df.shape}")
    hdf5.close()

    df['inst_det'] = df['gen_info.instrument'] + '/' + df['gen_info.detector']
    df['colormap'] = df['gen_info.detector']

    for key, value in DETECTOR_LEGEND.items():
        df.loc[df['gen_info.detector'] == key, 'colormap'] = value
    print(f"Detector colormaps added: {df.shape}") # (9804, 172)

    if save_csv:
        df['idx'] = df.index
        df.to_csv(save_csv, index=False)
        print(f"Dataframe saved to CSV: {save_csv}")

    return df


def make_subsets(df, column_groups, data_dir=None):
    subset_dict = {}
    for name in column_groups:
        df_name = name.lower()
        cols = [col for col in df if name in col]
        df_sub = df.loc[:, cols]
        subset_dict[df_name] = df_sub
        print(f"{name} > {df_name}: {len(cols)}")
        if data_dir:
            data_dir = data_dir.rstrip('/')
            csv_file = f'{data_dir}/{df_name}.csv'
            df_sub['idx'] = df_sub.index
            df_sub.to_csv(csv_file, index=False)
    print(f"{len(subset_dict)} subset DFs created.")
    return subset_dict


def extract_columns(df, datacols, header_cols, gen_info_cols):
    # for plotting
    header_cols = [col for col in df if 'header' in col]
    gen_info_cols = [col for col in df if 'gen_info' in col]
    columns = list(datacols) + header_cols + gen_info_cols + ['inst_det', 'colormap']
    extract_cols = list(set(columns))
    column_df = df.loc[:, extract_cols]
    return column_df


if __name__ == '__main__':
    #python -m import_hdf5.py ./data/2021-03-25/svm_2021-03-25.h5
    # args = sys.argv
    # if len(args) > 1:
    #     h5_file = args[-1]
    # else: #default
    #     h5_file = './data/2021-07-28/results_2021-07-28_svm_qa_dataframe.h5'
    # h5_file = './data/2021-03-25/results_2021-03-25_svm_qa_dataframe.h5'
    h5_file = './data/2021-07-28/results_2021-07-28_svm_qa_dataframe.h5'
    df = svm_harvester(h5_file, save_csv='./data/2021-07-28/dataframe.csv')
    column_groups = ['header', 'gen_info', 'number_of_sources', 'WCS', 'distribution', 'Cross-match', 'Segment', 'Delta', 'Statistics_', 'Interfilter', 'GAIA_sources']
    subset_dict = make_subsets(df, column_groups, data_dir='./data/2021-07-28')
    
    

    
