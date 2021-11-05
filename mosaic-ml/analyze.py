# STANDARD libraries
import os
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
font_dict={'family':'"Titillium Web", monospace','size':16}
mpl.rc('font',**font_dict)
import plotly.express as px
import plotly.graph_objs as go
from plotly import subplots
import plotly.offline as pyo
import plotly.figure_factory as ff
from keras.preprocessing import image


from augment import apply_power_transform



HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
saved_plots = f'{SUBFOLDER}/plots'
html_plots = os.path.join(saved_plots, 'html')

def plot_image_sets(X_tr, y_tr, X_vl, y_vl):
    posA = X_tr[-X_vl.shape[0]:][y_tr[-X_vl.shape[0]:]==1]
    posB = X_vl[y_vl==1]

    plt.figure(figsize=(10, 10))
    for n in range(5):
        x = image.array_to_img(posA[n][0])
        ax = plt.subplot(5, 5, n + 1)
        ax.imshow(x)
        plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    for n in range(5):
        x = image.array_to_img(posB[n][0])
        ax = plt.subplot(5, 5, n + 1)
        ax.imshow(x)
        plt.axis("off")
    plt.show()

def df_by_detector(df):
    hrc = df[df["dete_cat"] == 0.0]
    ir = df[df["dete_cat"] == 1.0]
    sbc = df[df["dete_cat"] == 2.0]
    uvis = df[df["dete_cat"] == 3.0]
    wfc = df[df["dete_cat"] == 4.0]
    det_dict = {
        "hrc": hrc,
        "ir": ir,
        "sbc": sbc,
        "uvis": uvis,
        "wfc": wfc
    }
    return det_dict

def make_scatter_figs(df, xaxis_name, yaxis_name, marker_size=15, c0='cyan', 
                      c1='fuchsia', detectors=True, show=True, save_html=None):
    if detectors is True:
        det_dict = df_by_detector(df)
    else:
        det_dict = {"all detectors": df}

    scatter_figs = []
    for detector, data in det_dict.items():
        miss = data.loc[data['label']==1]
        align = data.loc[data['label']==0]
        trace0 = go.Scatter(
            x=align[xaxis_name],
            y=align[yaxis_name],
            text=align.index,
            mode="markers",
            opacity=0.7,
            marker={"size": marker_size, "color": c0},
            name="aligned"  
        )
        trace1 = go.Scatter(
            x=miss[xaxis_name],
            y=miss[yaxis_name],
            text=miss.index,
            mode="markers",
            opacity=0.7,
            marker={"size": marker_size, "color": c1},
            name="misaligned"
        )
        traces=[trace0, trace1]
        layout = go.Layout(
            xaxis={"title": xaxis_name},
            yaxis={"title": yaxis_name},
            title=detector,
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode="closest",
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
            width=700,
            height=500
        )
        fig = go.Figure(data=traces, layout=layout)
        if show:
            fig.show()
        if save_html:
            if not os.path.exists(save_html):
                os.makedirs(save_html, exist_ok=True)
            pyo.plot(fig, filename=f"{save_html}/{detector}-{xaxis_name}-{yaxis_name}-scatter.html")
        scatter_figs.append(fig)
    return scatter_figs

def bar_plots(df, feature, errors=True, width=700, height=500, cmap=['dodgerblue', 'fuchsia'], 
              show=True, save_html=None):
    D, N = [0.0, 1.0, 2.0, 3.0, 4.0], ['hrc', 'ir', 'sbc', 'uvis', 'wfc']
    A = [df[(df["label"]==0) & (df['dete_cat']==d)][feature] for d in D]
    M = [df[(df["label"]==1) & (df['dete_cat']==d)][feature] for d in D]
    mu_0, mu_1 = [np.mean(a) for a in A], [np.mean(m) for m in M]
    if errors:
        e_0 = [np.std(a)/np.sqrt(len(a)) for a in A]
        e_1 = [np.std(m)/np.sqrt(len(m)) for m in M]
    else:
        e_0, e_1 = None, None
    trace0 = go.Bar(
            x=N,
            y=mu_0,
            error_y=dict(type="data", array=e_0, color="white", thickness=0.5),
            name="aligned",
            marker=dict(color=cmap[0]),
        )
    trace1 = go.Bar(
            x=N,
            y=mu_1,
            error_y=dict(type="data", array=e_1, color="white", thickness=0.5),
            name="misaligned",
            marker=dict(color=cmap[1]),
        )
    traces = [trace0, trace1]
    layout = go.Layout(
        title=f"Aligned vs Misaligned {feature}",
        xaxis={"title": "detector"},
        yaxis={"title": f"{feature} (mean)"},
        paper_bgcolor="#242a44",
        plot_bgcolor="#242a44",
        font={"color": "#ffffff"},
        width=width,
        height=height
    )
    fig = go.Figure(data=traces, layout=layout)
    if save_html:
        pyo.plot(fig, filename=f"{save_html}/{feature}-barplot.html")
    if show:
        fig.show()
    else:
        return fig


def kde_plots(df, cols, targets=False, hist=True, curve=True, binsize=None, width=700, height=500, cmap=None, show=True, save_html=html_plots):
    if targets is True:
        neg = df.loc[df['label'] == 0][cols[0]]
        pos = df.loc[df['label'] == 1][cols[0]]
        hist_data = [neg, pos]
        group_labels = [f'{cols[0]}=0', f'{cols[0]}=1']
        title = f'KDE {cols[0]}: aligned vs misaligned'
        name = f"kde-targets-{cols[0]}.html"
    else:
        hist_data = [df[c] for c in cols]
        group_labels = cols
        title = f'KDE {group_labels[0]} vs {group_labels[1]}'
        name = f"kde-{group_labels[0]}-{group_labels[1]}.html"
    if cmap is None:
        colors = ['#F66095', '#2BCDC1']
    else:
        colors = cmap
    if binsize is None:
        binsize = 0.2 #[0.3, 0.2, 0.1]
    fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=binsize, show_hist=hist, show_curve=curve)

    fig.update_layout(title_text=title, paper_bgcolor="#242a44",
        plot_bgcolor="#242a44",
        font={"color": "#ffffff"},
        width=width,
        height=height)
    if save_html:
        if not os.path.exists(save_html):
            os.makedirs(save_html, exist_ok=True)
        pyo.plot(fig, filename=f"{save_html}/{name}")
    if show:
        fig.show()
    return fig


def make_subplots(data1, data2, name1, name2, xtitle, ytitle, width=1300, height=700, show=True, save_html=html_plots):
    fig = subplots.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(name1, name2),
        shared_yaxes=False,
        x_title=xtitle,
        y_title=ytitle,
    )
    fig.add_trace(data1.data[0], 1, 1)
    fig.add_trace(data1.data[1], 1, 1)
    fig.add_trace(data2.data[0], 1, 2)
    fig.add_trace(data2.data[1], 1, 2)
    fig.update_layout(
        title_text=f"{name1} vs {name2}",
        margin=dict(t=50, l=80),
        width=width,
        height=height,
        paper_bgcolor="#242a44",
        plot_bgcolor="#242a44",
        font={
            "color": "#ffffff",
        },
    )
    if show:
        fig.show()
    if save_html:
        if not os.path.exists(save_html):
            os.makedirs(save_html, exist_ok=True)
        pyo.plot(fig, filename=f"{save_html}/kde_{name1}_vs_{name2}")
    return fig
        
# GROUPED BAR CHART
def grouped_barplot(df, save=False):
    groups = df.groupby(['dete_cat'])['label']
    hrc = groups.get_group(0.0).value_counts()
    ir = groups.get_group(1.0).value_counts()
    sbc = groups.get_group(2.0).value_counts()
    uvis = groups.get_group(3.0).value_counts()
    wfc = groups.get_group(4.0).value_counts()
    trace1 = go.Bar(
        x=hrc.index,
        y=hrc,
        name = 'HRC',
        marker = dict(color='red')
    )
    trace2 = go.Bar(
        x=ir.index,
        y=ir,
        name = 'IR',
        marker=dict(color='orange')
    )
    trace3 = go.Bar(
        x=sbc.index,
        y=sbc,
        name = 'SBC',
        marker = dict(color='yellow')
    )
    trace4 = go.Bar(
        x=uvis.index,
        y=uvis,
        name = 'UVIS',
        marker = dict(color='purple')
    )
    trace5 = go.Bar(
        x=wfc.index,
        y=wfc,
        name = 'WFC',
        marker = dict(color='blue')
    )
    data = [trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(
        title = 'SVM Alignment Labels by Detector'
    )
    fig = go.Figure(data=data, layout=layout)
    if save is True:
        pyo.plot(fig, filename='bar2.html')
    return fig


def raw_norm_plots(plot, df, df2, cols, xtitle, ytitle, targets=False, width=1300, height=700, show=True, save_html=None):
    if len(cols) == 1:
        title = f"{cols[0]} raw vs norm"
        name = f"{cols[0]}_rawnorm_{plot}"
    else:
        title = f"{cols[0]}-{cols[1]} raw vs norm"
        name = f"{cols[0]}_vs_{cols[1]}_rawnorm_{plot}"
    if plot == 'kde':
        data1 = kde_plots(df, cols, targets=targets, cmap=None, show=False, save_html=None)
        data2 = kde_plots(df2, cols, targets=targets, cmap=['cyan', 'fuchsia'], show=False, save_html=None)
    elif plot == 'scatter':
        data1 = make_scatter_figs(df, cols[0], cols[1], marker_size=7, c0='dodgerblue', c1='orange', detectors=False, save_html=None)
        data2 = make_scatter_figs(df2, cols[0]+'_scl', cols[1]+'_scl', marker_size=7, c0='cyan', c1='fuchsia', detectors=False, save_html=None)
    elif plot == 'bar':
        data1 = bar_plots(df, cols[0], cmap=['dodgerblue', 'orange'], show=False, save_html=None)
        data2 = bar_plots(df2, cols[0]+'_scl', cmap=['cyan', 'fuchsia'], show=False, save_html=None)
    fig = subplots.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('raw', 'norm'),
        shared_yaxes=False,
        x_title=xtitle,
        y_title=ytitle,
    )
    fig.add_trace(data1.data[0], 1, 1)
    fig.add_trace(data1.data[1], 1, 1)
    fig.add_trace(data2.data[0], 1, 2)
    fig.add_trace(data2.data[1], 1, 2)
    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=80),
        width=width,
        height=height,
        paper_bgcolor="#242a44",
        plot_bgcolor="#242a44",
        font={
            "color": "#ffffff",
        },
    )
    if show:
        fig.show()
    if save_html:
        if not os.path.exists(save_html):
            os.makedirs(save_html, exist_ok=True)
        pyo.plot(fig, filename=f"{save_html}/{name}.html")
    return fig


if __name__ == '__main__':
    df = pd.read_csv(f'{SUBFOLDER}/detection_cleaned.csv', index_col='index')
    df.drop(['category', 'ra_targ', 'dec_targ', 'imgname'], axis=1, inplace=True)
    rms_scatter = make_scatter_figs(df, 'rms_ra', 'rms_dec', detectors=True, save_html=html_plots)
    source_scatter = make_scatter_figs(df, 'point', 'segment', detectors=True, save_html=html_plots)
    cols = ['rms_ra', 'rms_dec', 'gaia_sources', 'nmatches', 'n_exposures']
    barplots = [bar_plots(df, c) for c in cols]
    df2, pt_transform = apply_power_transform(df)
    
    barplots_norm = [bar_plots(df2, c+'_scl') for c in cols]