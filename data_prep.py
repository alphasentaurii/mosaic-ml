import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from pprint import pprint
from tqdm import tqdm
import glob
import tensorflow as tf
from astropy.io import fits


HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
SCRUB = os.path.join(SUBFOLDER, 'scrubbed')
IMG_DIR = os.path.join(SUBFOLDER, 'images')
TRAIN_PATH = f"{IMG_DIR}/training"



"""REGRESSION TEST DATA PREP FOR MLP"""

def make_trees(dataset):
    tmp = dataset.copy()
    tree_dict = {}
    for idx, row in tmp.iterrows():
        if len(row['name']) > 6:
            tree_dict[idx] = 'leaf'
        elif row['filter'] == 'total':
            tree_dict[idx] = 'branch'
        else:
            tree_dict[idx] = 'stem'
    trees = pd.DataFrame.from_dict(tree_dict, orient='index', columns=['tree'])
    df = tmp.join(trees, how='left')
    return df

def split_index(df):
    idx_dict = {}
    for idx, _ in df.iterrows():
        n = str(idx)
        items = n.split('_')[1:]
        idx_dict[n] = {}
        idx_dict[n]['id'] = int(items[0])
        idx_dict[n]['visit'] = items[1]
        idx_dict[n]['instr'] = items[2]
        idx_dict[n]['detector'] = items[3]
        idx_dict[n]['filter'] = items[4]
        idx_dict[n]['branch'] = items[5]
        idx_dict[n]['dataset'] = items[5][:6]
    df_index = pd.DataFrame.from_dict(idx_dict, orient='index')
    #df_index['id'] = df_index['id'].astype('int')
    df = df_index.join(df, how='left')
    return df

def set_datetime_cols(df):
    if 'date-obs' in df.columns:
        df_obstime = df[['date-obs', 'time-obs']]
        isotimes = []
        for idx, row in df_obstime.iterrows():
            date_iso = row['date-obs']
            time_iso = row['time-obs']
            isotime = date_iso + 'T' + time_iso
            isotimes.append(isotime)

        dt_obs = pd.Series(isotimes, index=df.index, name='dt_obs')
        dt_obs = pd.to_datetime(dt_obs)

        dates, times = [], []
        for idx, row in dt_obs.items():
            d = pd.to_datetime(dt_obs[idx]).date()
            t = pd.to_datetime(dt_obs[idx]).time()
            dates.append(d)
            times.append(t)

        time_obs = pd.Series(times, index=df_obstime.index, name='time_obs')
        date_obs = pd.to_datetime(pd.Series(dates, index=df_obstime.index, name='date_obs'))
        df_dt = pd.concat([date_obs, time_obs], axis=1)
        df_obs = pd.concat([dt_obs, df_dt], axis=1)

        df_obs = df_obs.join(df, how='left')
        return df_obs
    else:
        return df

def rename_prefix_cols(df, splitter, prefix):
    columns = df.columns
    cols = [prefix+col.split(splitter)[-1].lower() for col in columns[1:]]
    cols.insert(0, prefix+'wcsname')
    hc = dict(zip(columns, cols))
    df.rename(hc, axis='columns', inplace=True)
    return df

def rename_wcs(df):
    wcs = pd.DataFrame(index=df.index)
    wcs_dict = {
        'Primary' : 'prime_',
        'AlternateWCS_default': 'alt_',
        'AlternateWCS_apriori': 'aprio_a_',
        'AlternateWCS_aposteriori': 'apost_a_',
        'DeltaWCS_default': 'delta_',
        'DeltaWCS_apriori': 'aprio_d_',
        'DeltaWCS_aposteriori': 'apost_d_'
    }
    for key, prefix in wcs_dict.items():
        key_cols = [col for col in df if key in col]
        pre = df.loc[:, key_cols]
        sub = rename_prefix_cols(pre, splitter='.', prefix=prefix)
        wcs = wcs.join(sub, how='left')
    return wcs

def rename_stat(df):
    stat_cols = [f"{col.split('.')[1].lower()}_{col.split('.')[-1].lower()}" for col in df.columns]
    hc = dict(zip(df.columns, stat_cols))
    df.rename(hc, axis='columns', inplace=True)
    return df

def rename_gaia(df):
    df.rename({'Number_of_GAIA_sources.Number_of_GAIA_sources': 'gaia_sources'}, axis=1, inplace=True)
    cols = [col.split('.')[-1].lower() for col in df.columns]
    hc = dict(zip(df.columns, cols))
    df.rename(hc, axis='columns', inplace=True)
    return df

def rename_filter(df):
    columns = df.columns
    xcentroid = [col for col in columns if 'xcentroid_ref_comparison' in col]
    xc = ['xc_'+col.split('.')[-1].lower() for col in xcentroid]
    ycentroid = [col for col in columns if 'ycentroid_ref_comparison' in col]
    yc = ['yc_'+col.split('.')[-1].lower() for col in ycentroid]
    refcat = [col for col in columns if 'cross-matched_reference_catalog' in col]
    rc = ['rc_'+col.split('.')[-1] for col in refcat]
    comcat = [col for col in columns if 'cross-matched_comparison_catalog' in col]
    cc = ['cc_'+col.split('.')[-1] for col in comcat]
    hc = dict(zip(xcentroid, xc))
    hc.update(dict(zip(ycentroid, yc)))
    hc.update(dict(zip(refcat, rc)))
    hc.update(dict(zip(comcat, cc)))
    other_cols = [col for col in list(columns) if col not in list(hc.keys())]
    cols = [col.split('.')[-1] for col in other_cols]
    hc.update(dict(zip(other_cols, cols)))
    df.rename(hc, axis='columns', inplace=True)
    return df

def rename_xmatch(df):
    columns = df.columns
    pnt = [col for col in columns if 'Cross-matched_point' in col]
    seg = [col for col in columns if 'Cross-matched_segment' in col]
    p = ['point_'+col.split('.')[-1].lower().replace(' ', '_') for col in pnt]
    s = ['segment_'+col.split('.')[-1].lower().replace(' ', '_') for col in seg]
    hc = dict(zip(pnt, p))
    hc.update(dict(zip(seg, s)))
    other_cols = [col for col in list(columns) if col not in list(hc.keys())]
    cols = [col.split('.')[-1] for col in other_cols]
    hc.update(dict(zip(other_cols, cols)))
    df = df.rename(hc, axis='columns', inplace=False)
    return df

def rename_cols(df, splitter='.', ops=None):
    if ops == 'wcs':
        df = rename_wcs(df)
    elif ops == 'stat':
        df = rename_stat(df)
    elif ops == 'gaia':
        df = rename_gaia(df)
    elif ops == 'filter':
        df = rename_filter(df)
    elif ops == 'xmatch':
        df = rename_xmatch(df)
    else:
        cols = [col.split(splitter)[-1].lower() for col in df.columns]
        hc = dict(zip(df.columns, cols))
        df.rename(hc, axis='columns', inplace=True)
    return df

def scrub_dataframe(data_file, drop_cols=None, drop_na=False, ops=None):
    df = pd.read_csv(f'{SUBFOLDER}/{data_file}', index_col='idx')
    df = rename_cols(df, ops=ops)
    df = set_datetime_cols(df) # header.csv
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    if drop_na:
        df.dropna(axis=0, inplace=True)
    df = split_index(df) # do this at the end for final df?
    return df

def clean_datasets():
    scrubbed_data = {}
    scrub_dict = {
        'gen_info.csv' : {
            'drop_cols': {'telescope', 'proposal_id', 'instrument', 'detector', 'visit', 'filter', 'dataset','dataframe_index', 'generation date', 'commit id', 'description', 'data_source'},
            'drop_na': False,
            'ops' : None
            },
        'header.csv' : {
            'drop_cols': {'mtflag', 'date-obs', 'time-obs', 'scan_typ', 'chinject'},
            'drop_na': False,
            'ops' : None
        },
        'number_of_sources.csv' : {
            'drop_cols': {'detector'},
            'drop_na': True,
            'ops' : None
            },
        'wcs.csv' : {
            'drop_cols': None,
            'drop_na': True,
            'ops': 'wcs'
        },
        'statistics_.csv' : {
            'drop_cols': None,
            'drop_na': True,
            'ops': 'stat'
        },
        'distribution.csv': {
            'drop_cols': None,
            'drop_na': True,
            'ops': None
        },
        'gaia_sources.csv': {
            'drop_cols': 'number_of_gaia_sources',
            'drop_na' : False,
            'ops': 'gaia'
        },
        'interfilter.csv': {
            'drop_cols': {'rc_xcentroid_ref', 'rc_ycentroid_ref', 'cc_xcentroid_ref', 'cc_ycentroid_ref', 'delta_xcentroid_ref', 'delta_ycentroid_ref'},
            'drop_na': True,
            'ops': 'filter'
        },
        'cross-match.csv': {
            'drop_cols': None,
            'drop_na': True,
            'ops': 'xmatch'
        }
    }
    for k, v in scrub_dict.items():
        df = scrub_dataframe(k, v['drop_cols'], v['drop_na'], v['ops'])
        df = make_trees(df)
        df['index'] = df.index
        df.to_csv(f'{SCRUB}/k', index=False)
        key = k.split('.')[0]
        scrubbed_data[key] = df
    return scrubbed_data

def cluster_image_labels(svm, keys, column):
    #keywords = ['GALAXY', 'GALAXY;GALAXY'] # + CLUSTER OF GALAXIES ?
    #keywords = [k.strip(' ') for k in keys]
    key_idx = []
    for idx, row in svm.iterrows():
        cat = row[column]
        split = cat.split('**')
        for s in split:
            if s.strip(' ') in keys:
                key_idx.append(idx)
    label_visits = list(svm.loc[svm.index.isin(key_idx)]['visit'])
    #print(len(label_visits))
    return label_visits

def encode_categories(df, category_dict=None):
    if category_dict is None:
        category_dict = {
            'O': ['EXT-CLUSTER', 'CALIBRATION', 'ISM', 'EXT-MEDIUM', 'CALIBRATION'],
            'U': ['UNIDENTIFIED', 'UNIDENTIFIED;SOLAR SYSTEM'],
            'SC': ['STELLAR CLUSTER', 'STELLARCLUSTER', 'CALIBRATION;STELLAR CLUSTER', 'CALIBRATION;STELLARCLUSTER', 'EXT-CLUSTER'],
            'S': ['STAR', 'EXT-STAR'],
            'GC': ['CLUSTER OF GALAXIES'],
            'G': ['GALAXY', 'GALAXY;GALAXY']
        }
    for k, v in category_dict.items():
        visits = cluster_image_labels(df, v, column='category')
        df['category'].loc[df.visit.isin(visits)] = k
    return df

def encode_wcs(df):
    df['wcs'] = 0
    df['wcs'].loc[df['wcstype']=='default'] = 1
    df['wcs'].loc[df['wcstype']=='a priori'] = 2
    df['wcs'].loc[df['wcstype']=='a posteriori'] = 3
    print(df['wcs'].value_counts())
    return df

def scrub_svm_labeled_data(svm):
    svm.drop('Column1', axis=1, inplace=True)
    # drop NaNs
    nans = svm.loc[(svm.index == 1211 )|( svm.index==1212) | (svm.index==1213)]
    svm.drop(nans.index, axis=0, inplace=True)
    # rename target class to 'label'
    svm['label'] = svm['success'].astype(int)
    svm.drop('success', axis=1, inplace=True)
    svm['label'].value_counts()
    # rename "config" to "detector" (and remove instrument since this is known by detector)
    dets = [c.split('/')[-1].lower() for c in svm['config']]
    svm['detector'] = dets
    # create index
    names = []
    for idx, row in svm.iterrows():
        dataset = row['visit']
        visitno = dataset[-2:]
        prop = int(row['proposal'])
        instr = row['config'].split('/')[0].lower()
        det = row['detector']
        name = f"hst_{prop}_{visitno}_{instr}_{det}_total_{dataset}"
        names.append(name)


    index = pd.Series(names)
    svm['index'] = index
    svm.set_index('index', drop=False, inplace=True)

    drops = ['target', 'dateobs', 'config', 'filter', 'aec', 'wcstype', 'wcsname', 'creation_date']
    svm = svm.drop(drops, axis=1, inplace=False)
    svm = encode_categories(svm)

def total_detection_merge(dfs, visit_cluster=None, verbose=0):
    H = dfs['header']
    I = dfs['info']
    G = dfs['gaia']
    S = dfs['sources']
    # Gather features needed for TOTAL DETECTION Images: 
    if visit_cluster is not None:
        D = H.loc[(H['dataset'].isin(visit_cluster)) & (H['tree']=='branch')]
    else:
        D = H.loc[H['tree']=='branch']

    index = D.index

    imgnames = I.loc[index]['imgname']
    D = D.join(imgnames, how='left')

    src = S.loc[index][['point', 'segment']]
    D = D.join(src, how='left')

    gaia = G.loc[index]['gaia_sources']
    D = D.join(gaia, how='left')

    # Recast bool/categorical types
    D['subarray'] = D['subarray'].astype('int')
    # FGSLOCK should go in here once we have more variable data:
    cats = ['detector', 'gyromode', 'aperture']
    encoder = LabelEncoder()
    for cat in cats:
        enc = encoder.fit_transform(D[cat])
        cat_enc = f'{cat[:4]}_cat'
        D[cat_enc] = enc
        if verbose:
            enc_idx = D[cat_enc].value_counts().index
            cat_idx = D[cat].value_counts().index
            print("embeddings: cat")
            pprint(dict(zip(enc_idx, cat_idx)))

    # Drop header columns we don't need for model
    drops = ['filter', 'name', 'dt_obs', 'date_obs', 'time_obs', 'fgslock', \
             'obstype', 'tree', 'id', 'visit', 'instr', 'detector', \
             'aperture', 'gyromode']
    D = D.drop(drops, axis=1, inplace=False)

    # Set index to image name
    # D.set_index('imgname', drop=True, inplace=True)

    return D

def get_detection_labels(td_data, svm_data):
    svm_data.drop('index', axis=1, inplace=True)
    df = svm_data.join(td_data, how='left')
    missing = df.loc[df['imgname'].isna()==True].index
    df = df.drop(missing, axis=0)
    df['gaia_sources'].fillna(value=0, axis=0, inplace=True)
    drops = ['filter', 'wcstype', 'detector', 'dataset', 'gyro_cat', 'aper_cat', 'exptime']
    df.drop(drops, axis=1, inplace=True)
    df.drop(['category', 'ra_targ', 'dec_targ', 'imgname'], axis=1, inplace=True)
    df['index'] = df.index
    df.to_csv(f'{SUBFOLDER}/detection_cleaned.csv', index=False)
    return df


def apply_power_transform(data, cols=['n_exposures', 'rms_ra', 'rms_dec', 
                                      'nmatches', 'point', 'segment', 
                                      'gaia_sources']):
    data_cont = data[cols]
    idx = data_cont.index
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    input_matrix = pt.transform(data_cont)
    lambdas = pt.lambdas_
    normalized = np.empty((len(data), len(cols)))
    mu, sig = [], []
    for i in range(len(cols)):
        v = input_matrix[:, i]
        m, s = np.mean(v), np.std(v)
        x = (v - m) / s
        normalized[:, i] = x
        mu.append(m)
        sig.append(s)
    pt_dict = {"lambdas": lambdas, "mu": np.array(mu), "sigma": np.array(sig)}
    pprint(pt_dict)
    newcols = [c+'_scl' for c in cols]
    df_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
    df = data.drop(cols, axis=1, inplace=False)
    df = df_norm.join(df, how='left')
    return df, pt_dict


def power_transform_matrix(data, pt_data):
    if type(data) == pd.DataFrame:
        data = data.values
    data_cont = data[:, :7]
    data_cat = data[:, -3:]
    nrows = data_cont.shape[0]
    ncols = data_cont.shape[1]
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    pt.lambdas_ = pt_data['lambdas']
    input_matrix = pt.transform(data_cont)
    normalized = np.empty((nrows, ncols))
    for i in range(data_cont.shape[1]):
        v = input_matrix[:, i]
        m = pt_data['mu'][i]
        s = pt_data['sigma'][i]
        x = (v - m) / s
        normalized[:, i] = x
    data_norm = np.concatenate((normalized, data_cat), axis=1)
    return data_norm
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

def laplacian_noise(x):
    return np.random.laplace(x)


def logistic_noise(x):
    return np.random.logistic(x)

    
def random_apply(func, x, p):
    r = tf.random.uniform([], minval=0, maxval=1) 
    if r < p:
        return func(x)
    else:
        return x


def augment_random_noise(x):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations need to be applied
    # randomly to impose translational invariance.
    x = random_apply(laplacian_noise, x, p=0.8)
    x = random_apply(logistic_noise, x, p=0.8)
    return x


def augment_random_integer(x):
    n = np.random.randint(-1, 3)
    if x < 1:
        x = np.abs((x+n))
    else:
        x += n
    return x


def augment_data(xi):
    """Randomly apply noise to continuous data
    """ 
    xi = np.array([augment_random_integer(xi[0]), 
                   augment_random_noise(xi[1]), 
                   augment_random_noise(xi[2]),
                   augment_random_integer(xi[3]),
                   augment_random_noise(xi[4]),
                   augment_random_noise(xi[5]),
                   augment_random_integer(xi[6]),
                   xi[7],
                   xi[8],
                   xi[9]])
    return xi


def training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val):
    xtr = np.empty(X_train.shape, dtype='float32')
    xts = np.empty(X_test.shape, dtype='float32')
    xvl = np.empty(X_val.shape, dtype='float32')
    X_train, X_test, X_val = X_train.values, X_test.values, X_val.values
    y_train, y_test, y_val = y_train.values, y_test.values, y_val.values
    for i in range(X_train.shape[0]):
        xtr[i] = augment_data(X_train[i])
    for i in range(X_test.shape[0]):
        xts[i] = augment_data(X_test[i])
    for i in range(X_val.shape[0]):
        xvl[i] = augment_data(X_val[i])
    X_train = np.concatenate([X_train, xtr, xts, xvl])
    y_train = np.concatenate([y_train, y_train, y_test, y_val])
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    return X_train, y_train




if __name__ == '__main__':
    os.makedirs(SCRUB, exist_ok=True)
    dfs = clean_datasets()
    svm = pd.read_csv(SUBFOLDER+'/svm_2021-07-28.csv', sep=' ')
    detector_dfs = dict(header=dfs['header'], info=dfs['info'], sources=dfs['sources'], gaia=dfs['gaia'])
    total_detection = total_detection_merge(detector_dfs, visit_cluster=None, verbose=0)
    df = get_detection_labels(total_detection, svm)
    df, pt_transform = apply_power_transform(df)
    X = df.drop('label', axis=1, inplace=False)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)
    X_train_1, y_train_1 = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    X_train_norm_1 = power_transform_matrix(X_train_1, pt_transform)
    X_test_norm_1 = power_transform_matrix(X_test, pt_transform)
    X_val_norm_1 = power_transform_matrix(X_val, pt_transform)


