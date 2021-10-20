import os
import argparse
import pandas as pd
import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits, ascii
from astropy.visualization import ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt

HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')

 
def point_flag_color(x):
    if x <= 1:
        return 'red', 'Flag <= 1'
    elif x <= 5:
        return 'green', '2 <= Flag <= 5'
    else:
        return None, None # 'yellow', 'Flag > 5'

def segment_flag_color(x):
    if x <= 1:
        return 'blue', 'Flag <= 1'
    elif x <= 5:
        return 'green', '2 <= Flag <= 5'
    else:
        return None, None # 'yellow', 'Flag > 5'


def draw_catalogs(input_fits, name, catalog):
    cpath = os.path.join(os.path.dirname(input_fits), 'cat', catalog)
    cname = f'{name}_{catalog}-cat.ecsv'
    cfile = os.path.join(cpath, cname)
    if os.path.exists(cfile):
        cat = ascii.read(cfile).to_pandas()
    if len(cat) > 0:
        if 'Flags' in cat.columns:
            cflags = cat['Flags']
        else:
            flagcols = [c for c in cat.columns if 'Flags' in c]
            if len(flagcols) > 0:
                flags = cat.loc[:,flagcols].fillna(100, axis=0, inplace=False).apply(min, axis=1)
                if catalog == 'point':
                    fcolor_ = flags.apply(point_flag_color)
                elif catalog == 'segment':
                    fcolor_ = flags.apply(segment_flag_color)
                fcolor = fcolor_.apply(lambda x: x[0]).values
            else:
                cat, fcolor_, fcolor = None, None, None
    else:
        cat, fcolor_, fcolor = None, None, None
    return cat, fcolor_, fcolor



def generate_gaia_cat(input_fits, output_img, figsize=(24,24)):
    """
    Opens fits files from local directory path to generate total detection and filter-level 
    drizzled images aligned to WCS with GAIA catalog overlay.
    Saves png file named using portion of original fits filename:
    'hst_11570_0a_wfc3_uvis_total_ib1f0a_drc.fits' -> hst_11570_0a_wfc3_uvis_total_ib1f0a.png
    **args**
    input_fits: path to fits files used to generate images
    output_img: where to save the pngs (path)
    **kwargs**
    Assumes path to gaia catalogs is: 'cat/gaia/{name}_flc_metawcs_all_GAIAeDR3_ref_cat.ecsv'
    where 'name' ~= 'hst_11360_61_wfc3_uvis'
    figsize: size to make the figures (default=(24,24))
    """
    store = os.path.join(input_fits, '.DS_Store')
    if os.path.exists(store):
        os.remove(store)
    fits_files = os.listdir(input_fits)
        
    for f in fits_files:
        name = f.split('.')[0][:-4]
        hfile = os.path.join(input_fits, f)
        ras, decs = np.ndarray((0,)), np.ndarray((0,))
        with fits.open(hfile) as ff:
            hdu = ff[1]
            wcs = WCS(hdu.header)
            footprint = wcs.calc_footprint(hdu.header)
            ras = np.append(ras, footprint[:, 0])
            decs = np.append(decs, footprint[:, 1])
            ralim = [np.max(ras), np.min(ras)]
            declim = [np.max(decs), np.min(decs)]
            radeclim = np.stack([ralim, declim], axis=1)
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=wcs, frameon=False)
            plt.axis(False)
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(hdu.data)

            norm = ImageNormalize(hdu.data, vmin=0, vmax=vmax*2,
                                    clip=True)
            xlim, ylim = wcs.wcs_world2pix(radeclim, 1).T
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.imshow(hdu.data, origin='lower', norm=norm, cmap='gray')
        
            pfx = '_'.join(name.split('_')[:5])
            sfx = pfx.split('_')[-1]
            
            g_cat = os.path.join(os.path.dirname(input_fits), 'cat', 'gaia')
            if f.split('_')[-1] == 'drz.fits':
                gname = f'{pfx}_flt_metawcs_all_GAIAeDR3_ref_cat.ecsv'
            else:
                gname = f'{pfx}_flc_metawcs_all_GAIAeDR3_ref_cat.ecsv'
            gfile = os.path.join(g_cat, gname)
            if os.path.exists(gfile):
                gaia = ascii.read(gfile).to_pandas()
                ax.scatter(gaia['RA'], gaia['DEC'],
                edgecolor='cyan', facecolor='none',
                transform=ax.get_transform('fk5'),
                marker='o', s=15)
        out = f'{output_img}/{name}'
        os.makedirs(out, exist_ok=True)
        plt.savefig(os.path.join(out, f'{name}_gaia'), bbox_inches='tight')
        plt.close(fig)

def draw_total_images(input_fits, output_img, catalogs=[], figsize=(24,24), cmap='gray'):
    """
    Opens fits files from local directory path to generate total detection and filter-level 
    drizzled images aligned to WCS and point/segment/both/none catalog overlay options.
    Saves png file named using portion of original fits filename:
    'hst_11570_0a_wfc3_uvis_total_ib1f0a_drc.fits' -> hst_11570_0a_wfc3_uvis_total_ib1f0a.png
    **args**
    input_fits: path to fits files used to generate images
    output_img: where to save the pngs (path)
    **kwargs**
    catalogs: 'point' or 'segment' or both
    figsize: size to make the figures (default=(24,24))
    cmap: colormap to use for plot (default is 'gray')
    """
    if catalogs == ['source']:
        catalogs = ['point', 'segment']
    store = os.path.join(input_fits, '.DS_Store')
    if os.path.exists(store):
        os.remove(store)
    fits_files = os.listdir(input_fits)
        
    for f in fits_files:
        name = f.split('.')[0][:-4]
        hfile = os.path.join(input_fits, f)
        ras, decs = np.ndarray((0,)), np.ndarray((0,))
        with fits.open(hfile) as ff:
            hdu = ff[1]
            wcs = WCS(hdu.header)
            footprint = wcs.calc_footprint(hdu.header)
            ras = np.append(ras, footprint[:, 0])
            decs = np.append(decs, footprint[:, 1])
            ralim = [np.max(ras), np.min(ras)]
            declim = [np.max(decs), np.min(decs)]
            radeclim = np.stack([ralim, declim], axis=1)
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=wcs, frameon=False)
            plt.axis(False)
            interval = ZScaleInterval()
            _, vmax = interval.get_limits(hdu.data)

            norm = ImageNormalize(hdu.data, vmin=0, vmax=vmax*2,
                                    clip=True)
                
            ax.imshow(hdu.data, origin='lower', norm=norm, cmap=cmap)

        if 'point' in catalogs:
            point, pfcolor_, pfcolor = draw_catalogs(input_fits, name, 'point')
            if point is not None:
                for fcol in pfcolor_.unique():
                    if fcol is not None:
                        q = pfcolor == fcol[0]
                        ax.scatter(point[q]['RA'], point[q]['DEC'],
                                    edgecolor=fcol[0], facecolor='none',
                                    transform=ax.get_transform('fk5'),
                                    marker='o', s=15, alpha=0.5)

        if 'segment' in catalogs:
            seg, sfcolor_, sfcolor = draw_catalogs(input_fits, name, 'segment')
            if seg is not None:
                for fcol in sfcolor_.unique():
                    if fcol is not None:
                        q = sfcolor == fcol[0]
                        ax.scatter(seg[q]['RA'], seg[q]['DEC'],
                                edgecolor=fcol[0], facecolor='none',
                                transform=ax.get_transform('fk5'),
                                marker='o', s=15, alpha=0.5)

        xlim, ylim = wcs.wcs_world2pix(radeclim, 1).T
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if len(catalogs)>1:
            catstr = '_source'
        elif len(catalogs) == 1:
            catstr = f'_{catalogs[0]}'
        else:
            if cmap == 'gray':
                catstr = ''
            else:
                catstr = f'_{cmap.lower()}'
        out = f'{output_img}/{name}'
        os.makedirs(out, exist_ok=True)
        plt.savefig(os.path.join(out, f'{name}{catstr}'), bbox_inches='tight')
        plt.close(fig)


def get_filter_files(df, filter_path, dete_cat):
    input_filters = os.path.join(filter_path, 'fits')
    total = df.loc[df['dete_cat'] == dete_cat]
    datasets = total['dataset']
    filter_files = {}
    for data in datasets:
        filter_files[data] = []
        for file in os.listdir(input_filters):
            f = file.split('_')[-2]
            if f == data:
                filter_files[data].append(file)
    return filter_files


def draw_filter_images(filter_files, filter_path, figsize=(24,24)):
    input_fits = os.path.join(filter_path, 'fits')
    output_img = os.path.join(filter_path, 'img')
    os.makedirs(output_img, exist_ok=True)
    store = os.path.join(input_fits, '.DS_Store')
    if os.path.exists(store):
        os.remove(store)
    for dataset, filenames in filter_files.items():
        for filename in filenames:
            hfile = os.path.join(input_fits, filename)
            if os.path.exists(hfile):
                ras, decs = np.ndarray((0,)), np.ndarray((0,))
                name = filename.split('.')[0][:-4]
                with fits.open(hfile) as ff:
                    hdu = ff[1]
                    wcs = WCS(hdu.header)
                    footprint = wcs.calc_footprint(hdu.header)
                    ras = np.append(ras, footprint[:, 0])
                    decs = np.append(decs, footprint[:, 1])
                    ralim = [np.max(ras), np.min(ras)]
                    declim = [np.max(decs), np.min(decs)]
                    radeclim = np.stack([ralim, declim], axis=1)

                    fig = plt.figure(figsize=figsize, edgecolor='k', frameon=False)
                    ax = fig.add_subplot(111, projection=wcs, frameon=False)
                    plt.axis(False)
                    interval = ZScaleInterval()
                    vmin, vmax = interval.get_limits(hdu.data)
                    norm = ImageNormalize(hdu.data, vmin=vmin, vmax=vmax*2,
                                            clip=True)
                    xlim, ylim = wcs.wcs_world2pix(radeclim, 1)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.imshow(hdu.data, origin='lower', norm=norm, cmap='gray')

                out = f'{output_img}/{dataset}'
                os.makedirs(out, exist_ok=True)
                plt.savefig(os.path.join(out, f'{name}'), bbox_inches='tight')
                plt.close(fig)

def generate_total_images(input_fits, output_img=None, catalog=None, generator=0):
    if output_img is None:
        output_img = os.path.join(os.path.dirname(input_fits), 'img')
    if generator == 1:
        draw_total_images(input_fits, output_img, cmap='gray')
        draw_total_images(input_fits, output_img, catalogs=['segment'], cmap='gray')
        draw_total_images(input_fits, output_img, catalogs=['point'], cmap='gray')
        draw_total_images(input_fits, output_img, catalogs=['point', 'segment'], cmap='gray')
        generate_gaia_cat(input_fits, output_img, figsize=(24,24))
    elif catalog is not None:
        if catalog == 'gaia':
            generate_gaia_cat(input_fits, output_img, figsize=(24,24))
        else:
            draw_total_images(input_fits, output_img, catalogs=[catalog], figsize=(24,24), cmap='gray')
    else:
        draw_total_images(input_fits, output_img, cmap='gray')

def generate_filter_images(input_fits, dataframe, detector):
    detectors = {'hrc': 0, 'ir': 1, 'sbc': 2, 'uvis': 3, 'wfc': 4}
    df = pd.read_csv(dataframe, index_col='index')
    if detector == 'all':
        for d, n in detectors.items():
            filter_path = os.path.join(input_fits, d)
            filter_files = get_filter_files(df, filter_path, n)
            draw_filter_images(filter_files, filter_path, figsize=(24,24))
    else:
        filter_files = get_filter_files(df, input_fits, detectors[detector])
        draw_filter_images(filter_files, input_fits, figsize=(24,24))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fits", type=str, help="path to fits file directory")
    parser.add_argument("-o", "--output", type=str, help="path to png image directory")
    parser.add_argument("-i", "--imagetype", type=str, choices=['total', 'filter'], help="draw total detection or filter level images")
    parser.add_argument("-c", "--catalog", type=str, choices=['point', 'segment', 'source', 'gaia'], help="make source catalog images. Use generator (-g) to generate all")
    parser.add_argument("-g", "--generator", type=int, choices=[0, 1], help="0: generate original only or 1: all types (original, point, segment, point-segment, gaia. For catalogs use -c flag")
    parser.add_argument("-f", "--datafile", type=str, help="path to total detection csv file.")
    parser.add_argument("-d", "--detector", type=str, choices=['hrc', 'ir', 'sbc', 'uvis', 'wfc', 'all'], help="detector name for filter images")
    args = parser.parse_args()
    input_fits = args.input_fits
    image_type = args.imagetype
    output_img = args.output
    if image_type == 'total':
        catalog = args.catalog
        generator = args.generator
        generate_total_images(input_fits, output_img, catalog, generator)
    else:
        dataframe = args.datafile
        detector = args.detector
        generate_filter_images(input_fits, dataframe, detector)

    
    

