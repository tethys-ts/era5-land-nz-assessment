"""
Created by Mike Kittridge on 2020-09-01.
Contains the code to train and test the flood forecast model.

"""
import os
import numpy as np
import pandas as pd
import requests
import zstandard as zstd
import pickle
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
# from sklearn.inspection import permutation_importance
from scipy.signal import argrelextrema
# %matplotlib inline
import orjson
import xarray as xr
from tethysts import Tethys
import yaml
import tethys_utils as tu
from shapely import wkb
from shapely.geometry import mapping
import geopandas as gpd

#####################################
### Parameters

base_path = os.path.realpath(os.path.dirname(__file__))

# with open(os.path.join(base_path, 'parameters.yml')) as param:
#     param = yaml.safe_load(param)

# source = param['source']

# public_url = 'https://b2.tethys-ts.xyz'

min_year_range = 40

islands_gpkg = 'islands.gpkg'

#####################################
### Functions


def create_shifted_df(series, from_range, to_range, freq_code, agg_fun, ref_name, include_0=False, discrete=False, **kwargs):
    """

    """
    if not isinstance(series, pd.Series):
        raise TypeError('series must be a pandas Series.')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('The series index must be a pandas DatetimeIndex.')

    df = series.reset_index()
    data_col = df.columns[1]
    ts_col = df.columns[0]
    s2 = tu.grp_ts_agg(df, None, ts_col, freq_code, agg_fun, discrete, **kwargs)[data_col]

    if include_0:
        f_hours = list(range(from_range-1, to_range+1))
        f_hours[0] = 0
    else:
        f_hours = list(range(from_range, to_range+1))

    df_list = []
    for d in f_hours:
        n1 = s2.shift(d, 'H')
        n1.name = ref_name + '_' + str(d)
        df_list.append(n1)
    data = pd.concat(df_list, axis=1).dropna()

    return data


####################################
### Get data

islands = gpd.read_file(os.path.join(base_path, islands_gpkg))

# TODO: Remove filter below to add in the north island
# islands = islands[islands.island == 'south'].copy()

## Datasets
tethys1 = Tethys()

datasets = tethys1.datasets.copy()

p_datasets1 = [d for d in datasets if (d['feature'] == 'atmosphere') and (d['parameter'] == 'precipitation') and (d['product_code'] == 'quality_controlled_data') and (d['frequency_interval'] == '1H')]
p_datasets2 = [d for d in datasets if (d['feature'] == 'atmosphere') and (d['parameter'] == 'precipitation') and (d['product_code'] == 'raw_data') and (d['frequency_interval'] == '1H') and (d['owner'] == 'FENZ')]

p_datasets = p_datasets1 + p_datasets2

era5_dataset = [d for d in datasets if (d['feature'] == 'atmosphere') and (d['parameter'] == 'precipitation') and (d['product_code'] == 'reanalysis-era5-land') and (d['frequency_interval'] == 'H')][0]


## Stations
p_stns = []

for d in p_datasets:
    for island in islands.island:
        poly = islands[islands.island == island].geometry.iloc[0]
        poly_geo = mapping(poly)

        p_stns1 = tethys1.get_stations(d['dataset_id'], geometry=poly_geo)
        [s.update({'island': island}) for s in p_stns1]
        p_stns.extend(p_stns1)

# Filter
p_stns2 = []

for s in p_stns:
    from_date = pd.to_datetime(s['time_range']['from_date'])
    to_date = pd.to_datetime(s['time_range']['to_date'])
    year_range = int((to_date - from_date).days/365)
    if year_range >= min_year_range:
        p_stns2.append(s)

era5_stns = []
for island in islands.island:
    poly = islands[islands.island == island].geometry.iloc[0]
    poly_geo = mapping(poly)

    p_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)
    [s.update({'island': island}) for s in p_stns1]
    era5_stns.extend(p_stns1)

era5_stn_ids = [s['station_id'] for s in era5_stns]

## TS Data
p_ds_ids = set([s['dataset_id'] for s in p_stns2])

p_data_list = []

for ds_id in p_ds_ids:
    stns = [s for s in p_stns2 if s['dataset_id'] == ds_id]
    stn_ids = [s['station_id'] for s in stns]
    p_data1 = tethys1.get_bulk_results(ds_id, stn_ids, squeeze_dims=True)
    for g in p_data1.geometry:
        g1 = str(g.values)
        geo = wkb.loads(g1, hex=True)
        val = p_data1.sel(geometry=g1)
        stn_id = str(val.station_id.values)
        island = [s['island'] for s in stns if s['station_id'] == stn_id][0]
        val2 = val['precipitation'].to_dataframe().reset_index().dropna()
        times = val2['time'][-5:-1]
        freq = pd.infer_freq(times)
        if freq == 'H':
            val3 = val2[['time', 'precipitation']].copy()
            val3['geo'] = geo
            val3['island'] = island
            val3['lat'] = geo.y
            val3['lon'] = geo.x
            val3['station_id'] = str(val.station_id.values)
            p_data_list.append(val3)

p_data = pd.concat(p_data_list)

p_stns3 = p_data.drop(['precipitation', 'time'], axis=1).drop_duplicates(['station_id'])

## ERA5 data

comp_list = []
for i, s in p_stns3.iterrows():
    print(s)

    poly_geo = mapping(s.geo.buffer(0.15))

    p_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)

    stn_ids = [s1['station_id'] for s1 in p_stns1]

    era1 = tethys1.get_bulk_results(era5_dataset['dataset_id'], stn_ids, squeeze_dims=True, cache='memory').dropna('time')

    era2 = era1.drop('height').to_dataframe().reset_index().drop(['lat', 'lon'], axis=1)
    era3 = era2.set_index(['station_id', 'time'])['precipitation'].unstack(0)

    # p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time').rename(columns={'precipitation': s['station_id']}).copy()
    p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time')['precipitation'].copy()

    era4 = era3[era3.index.isin(p_data2.index)].copy()
    p_data3 = p_data2[p_data2.index.isin(era4.index)].copy()

    if not p_data3.empty:

        ## Correct for data that is not hourly...
        r1 = p_data3.rolling(5, center=True)

        r2 = [pd.infer_freq(r.index) for r in r1]

        r3 = pd.Series(r2, index=p_data3.index)
        r3.loc[r3.isnull()] = 'Y'
        r3.loc[r3.str.contains('H')] = 'H'
        r3.loc[~(r3.str.contains('H') | r3.str.contains('Y'))] = 'D'
        r3.loc[r3.str.contains('Y')] = np.nan
        r3 = r3.fillna('ffill')
        r4 = r3 == 'H'

        p_data3 = p_data3[r4].copy()
        era4 = era4[era4.index.isin(p_data3.index)].copy()

        p_data4 = p_data3.resample('D').sum()
        era5 = era4.resample('D').sum()

        shift = [-1, 0, 1]

        ## Shift times in era5
        df_list = []
        for c in era5:
            s2 = era5[c]
            for d in shift:
                n1 = s2.shift(d, 'D')
                n1.name = c + '_' + str(d)
                df_list.append(n1)
        era6 = pd.concat(df_list, axis=1).dropna()

        p_data5 = p_data4[p_data4.index.isin(era6.index)].copy()

        from_date = p_data5.index[0]
        to_date = p_data5.index[-1]

        time_range = (to_date - from_date).days
        year_range = int(time_range/365)

        ## Package up for analysis
        if year_range >= 10:
            decades = year_range//10

            test_features_df = era6
            test_features = np.array(test_features_df)

            test_labels_df = p_data5
            test_labels = np.array(test_labels_df)

            results_list = []

            for i in range(decades):
                y = (i+1)*10
                start_date = to_date - pd.DateOffset(years=y)

                train_features_df = era6.loc[start_date:to_date]
                train_features = np.array(train_features_df)
                train_labels_df = p_data5.loc[start_date:to_date]
                train_labels = np.array(train_labels_df)

                # gbsq = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.1)
                # gbp = HistGradientBoostingRegressor(loss='poisson', max_iter=100, learning_rate=0.1)
                rfr = RandomForestRegressor(n_estimators = 200, n_jobs=4)
                rfc = RandomForestClassifier(n_estimators = 200, n_jobs=4)

                # model_dict = {'gbsq': gbsq, 'gbp': gbp, 'rfr': rfr}
                model_dict = {'rfr': rfr, 'rfc': rfc}

                for name, m in model_dict.items():
                    if name == 'rfc':
                        train_labels_c = (train_labels > 0.5).astype(int)
                        m.fit(train_features, train_labels_c)
                    else:
                        m.fit(train_features, train_labels)

                    ## Make the predictions and combine with the actuals
                    predictions1 = m.predict(test_features)

                    predict1 = pd.Series(predictions1, index=test_features_df.index, name='predicted')
                    predict1.loc[predict1 < 0] = 0

                    # if name == 'gbp':
                    #     predict1.loc[predict1 == predict1.min()] = 0

                    if name == 'rfc':
                        combo1 = pd.merge((test_labels_df > 0.5).astype(int).reset_index(), predict1.reset_index(), on='time', how='left').set_index('time')
                    else:
                        combo1 = pd.merge(test_labels_df.reset_index(), predict1.reset_index(), on='time', how='left').set_index('time')

                    combo1['error'] = combo1['predicted'] - combo1['precipitation']
                    combo1['AE'] = combo1['error'].abs()
                    mean_actual = combo1['precipitation'].mean()
                    mean_ae = combo1['AE'].mean()
                    nae = mean_ae/mean_actual
                    mean_error = combo1['error'].mean()
                    bias = mean_error/mean_actual

                    out_list = [s['station_id'], name, start_date, to_date, y, nae, bias]
                    out1 = pd.Series(out_list, index=['station_id', 'model', 'start', 'end', 'n_years', 'NAE', 'bias'])
                    out1.name = y

                    out2 = out1.to_frame().T.set_index(['station_id', 'n_years', 'model'])

                    results_list.append(out2)

            results1 = pd.concat(results_list)

        comp_list.append(results1)

comp1 = pd.concat(comp_list)

comp2 = comp1.groupby(level=['model', 'n_years'])[['NAE', 'bias']].mean()
comp2a = comp1.groupby(level=['model', 'n_years'])[['NAE', 'bias']].std()
comp1.groupby(level=['model', 'n_years'])[['NAE', 'bias']].count()

k
#################################
### Combo

comp_listb = []
for i, s in p_stns3.iterrows():
    print(s)

    poly_geo = mapping(s.geo.buffer(0.15))

    p_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)

    stn_ids = [s1['station_id'] for s1 in p_stns1]

    era1 = tethys1.get_bulk_results(era5_dataset['dataset_id'], stn_ids, squeeze_dims=True, cache='memory').dropna('time')

    era2 = era1.drop('height').to_dataframe().reset_index().drop(['lat', 'lon'], axis=1)
    era3 = era2.set_index(['station_id', 'time'])['precipitation'].unstack(0)

    # p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time').rename(columns={'precipitation': s['station_id']}).copy()
    p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time')['precipitation'].copy()

    era4 = era3[era3.index.isin(p_data2.index)].copy()
    p_data3 = p_data2[p_data2.index.isin(era4.index)].copy()

    if not p_data3.empty:

        ## Correct for data that is not hourly...
        r1 = p_data3.rolling(5, center=True)

        r2 = [pd.infer_freq(r.index) for r in r1]

        r3 = pd.Series(r2, index=p_data3.index)
        r3.loc[r3.isnull()] = 'Y'
        r3.loc[r3.str.contains('H')] = 'H'
        r3.loc[~(r3.str.contains('H') | r3.str.contains('Y'))] = 'D'
        r3.loc[r3.str.contains('Y')] = np.nan
        r3 = r3.fillna('ffill')
        r4 = r3 == 'H'

        p_data3 = p_data3[r4].copy()
        era4 = era4[era4.index.isin(p_data3.index)].copy()

        p_data4 = p_data3.resample('D').sum()
        era5 = era4.resample('D').sum()

        shift = [-1, 0, 1]

        ## Shift times in era5
        df_list = []
        for c in era5:
            s2 = era5[c]
            for d in shift:
                n1 = s2.shift(d, 'D')
                n1.name = c + '_' + str(d)
                df_list.append(n1)
        era6 = pd.concat(df_list, axis=1).dropna()

        p_data5 = p_data4[p_data4.index.isin(era6.index)].copy()

        from_date = p_data5.index[0]
        to_date = p_data5.index[-1]

        time_range = (to_date - from_date).days
        year_range = int(time_range/365)

        ## Package up for analysis
        if year_range >= 10:
            decades = year_range//10

            test_features_df = era6
            test_features = np.array(test_features_df)

            test_labels_df = p_data5
            test_labels = np.array(test_labels_df)

            results_list = []

            for i in range(decades):
                y = (i+1)*10
                start_date = to_date - pd.DateOffset(years=y)

                train_features_df = era6.loc[start_date:to_date]
                train_features = np.array(train_features_df)
                train_labels_df = p_data5.loc[start_date:to_date]
                train_labels = np.array(train_labels_df)

                # gbsq = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.1)
                # gbp = HistGradientBoostingRegressor(loss='poisson', max_iter=100, learning_rate=0.1)
                rfr = RandomForestRegressor(n_estimators = 200, n_jobs=4)
                rfc = RandomForestClassifier(n_estimators = 200, n_jobs=4)

                # model_dict = {'gbsq': gbsq, 'gbp': gbp, 'rfr': rfr}
                # model_dict = {'rfr': rfr, 'rfc': rfc}

                train_labels_c = (train_labels > 0.5).astype(int)
                rfc.fit(train_features, train_labels_c)
                rfr.fit(train_features, train_labels)

                ## Make the predictions and combine with the actuals
                predictions1 = rfc.predict(test_features)
                predictions2 = rfr.predict(test_features)

                predict1 = pd.Series(predictions1, index=test_features_df.index, name='predicted')
                predict1.loc[predict1 < 0] = 0
                # predict1 = predict1.astype(bool)

                predict2 = pd.Series(predictions2, index=test_features_df.index, name='predicted')
                predict2.loc[predict2 < 0] = 0

                predict3 = predict1 * predict2

                # if name == 'gbp':
                #     predict1.loc[predict1 == predict1.min()] = 0

                combo1 = pd.merge(test_labels_df.reset_index(), predict3.reset_index(), on='time', how='left').set_index('time')

                combo1['error'] = combo1['predicted'] - combo1['precipitation']
                combo1['AE'] = combo1['error'].abs()
                mean_actual = combo1['precipitation'].mean()
                mean_ae = combo1['AE'].mean()
                nae = mean_ae/mean_actual
                mean_error = combo1['error'].mean()
                bias = mean_error/mean_actual

                out_list = [s['station_id'], 'combo', start_date, to_date, y, nae, bias]
                out1 = pd.Series(out_list, index=['station_id', 'model', 'start', 'end', 'n_years', 'NAE', 'bias'])
                out1.name = y

                out2 = out1.to_frame().T.set_index(['station_id', 'n_years', 'model'])

                results_list.append(out2)

            results1 = pd.concat(results_list)

        comp_listb.append(results1)

comp3 = pd.concat(comp_listb)

comp4a = comp3.groupby(level=['model', 'n_years'])[['NAE', 'bias']].mean()
comp4b = comp3.groupby(level=['model', 'n_years'])[['NAE', 'bias']].std()
comp3.groupby(level=['model', 'n_years'])[['NAE', 'bias']].count()



### GB

comp_listc = []
for i, s in p_stns3.iterrows():
    print(s)

    poly_geo = mapping(s.geo.buffer(0.15))

    p_stns1 = tethys1.get_stations(era5_dataset['dataset_id'], geometry=poly_geo)

    stn_ids = [s1['station_id'] for s1 in p_stns1]

    era1 = tethys1.get_bulk_results(era5_dataset['dataset_id'], stn_ids, squeeze_dims=True, cache='memory').dropna('time')

    era2 = era1.drop('height').to_dataframe().reset_index().drop(['lat', 'lon'], axis=1)
    era3 = era2.set_index(['station_id', 'time'])['precipitation'].unstack(0)

    # p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time').rename(columns={'precipitation': s['station_id']}).copy()
    p_data2 = p_data[p_data.station_id == s['station_id']][['time', 'precipitation']].set_index('time')['precipitation'].copy()

    era4 = era3[era3.index.isin(p_data2.index)].copy()
    p_data3 = p_data2[p_data2.index.isin(era4.index)].copy()

    if not p_data3.empty:

        ## Correct for data that is not hourly...
        r1 = p_data3.rolling(5, center=True)

        r2 = [pd.infer_freq(r.index) for r in r1]

        r3 = pd.Series(r2, index=p_data3.index)
        r3.loc[r3.isnull()] = 'Y'
        r3.loc[r3.str.contains('H')] = 'H'
        r3.loc[~(r3.str.contains('H') | r3.str.contains('Y'))] = 'D'
        r3.loc[r3.str.contains('Y')] = np.nan
        r3 = r3.fillna('ffill')
        r4 = r3 == 'H'

        p_data3 = p_data3[r4].copy()
        era4 = era4[era4.index.isin(p_data3.index)].copy()

        p_data4 = p_data3.resample('D').sum()
        era5 = era4.resample('D').sum()

        shift = [-1, 0, 1]

        ## Shift times in era5
        df_list = []
        for c in era5:
            s2 = era5[c]
            for d in shift:
                n1 = s2.shift(d, 'D')
                n1.name = c + '_' + str(d)
                df_list.append(n1)
        era6 = pd.concat(df_list, axis=1).dropna()

        p_data5 = p_data4[p_data4.index.isin(era6.index)].copy()

        from_date = p_data5.index[0]
        to_date = p_data5.index[-1]

        time_range = (to_date - from_date).days
        year_range = int(time_range/365)

        ## Package up for analysis
        if year_range >= 10:
            decades = year_range//10

            test_features_df = era6
            test_features = np.array(test_features_df)

            test_labels_df = p_data5
            test_labels = np.array(test_labels_df)

            results_list = []

            for i in range(decades):
                y = (i+1)*10
                start_date = to_date - pd.DateOffset(years=y)

                train_features_df = era6.loc[start_date:to_date]
                train_features = np.array(train_features_df)
                train_labels_df = p_data5.loc[start_date:to_date]
                train_labels = np.array(train_labels_df)

                gbsq = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.1)
                # gbp = HistGradientBoostingRegressor(loss='poisson', max_iter=100, learning_rate=0.1)
                # rfr = RandomForestRegressor(n_estimators = 200, n_jobs=4)
                # rfc = RandomForestClassifier(n_estimators = 200, n_jobs=4)

                # model_dict = {'gbsq': gbsq, 'gbp': gbp, 'rfr': rfr}
                # model_dict = {'rfr': rfr, 'rfc': rfc}

                # train_labels_c = (train_labels > 0.5).astype(int)
                # gbsq.fit(train_features, train_labels_c)
                gbsq.fit(train_features, train_labels)

                ## Make the predictions and combine with the actuals
                predictions1 = gbsq.predict(test_features)
                # predictions2 = rfr.predict(test_features)

                predict1 = pd.Series(predictions1, index=test_features_df.index, name='predicted')
                predict1.loc[predict1 < 0] = 0
                # predict1 = predict1.astype(bool)

                # predict2 = pd.Series(predictions2, index=test_features_df.index, name='predicted')
                # predict2.loc[predict2 < 0] = 0

                # predict3 = predict1 * predict2

                # if name == 'gbp':
                #     predict1.loc[predict1 == predict1.min()] = 0

                combo1 = pd.merge(test_labels_df.reset_index(), predict1.reset_index(), on='time', how='left').set_index('time')

                combo1['error'] = combo1['predicted'] - combo1['precipitation']
                combo1['AE'] = combo1['error'].abs()
                mean_actual = combo1['precipitation'].mean()
                mean_ae = combo1['AE'].mean()
                nae = mean_ae/mean_actual
                mean_error = combo1['error'].mean()
                bias = mean_error/mean_actual

                out_list = [s['station_id'], 'HGB', start_date, to_date, y, nae, bias]
                out1 = pd.Series(out_list, index=['station_id', 'model', 'start', 'end', 'n_years', 'NAE', 'bias'])
                out1.name = y

                out2 = out1.to_frame().T.set_index(['station_id', 'n_years', 'model'])

                results_list.append(out2)

            results1 = pd.concat(results_list)

        comp_listc.append(results1)

comp5 = pd.concat(comp_listc)

comp6a = comp5.groupby(level=['model', 'n_years'])[['NAE', 'bias']].mean()
comp6b = comp5.groupby(level=['model', 'n_years'])[['NAE', 'bias']].std()
comp5.groupby(level=['model', 'n_years'])[['NAE', 'bias']].count()



















################################
### Other






wl_stn_id = [p['station_id'] for p in wl_sites][0]
wl_data1 = tethys1.get_results(wl_dataset['dataset_id'], wl_stn_id, remove_height=True)
ref = wl_sites[0]['ref']
wl_data1 = wl_data1.to_dataframe()['gage_height']
wl_data2 = wl_data1.resample('H').mean().interpolate('pchip', limit=24).dropna()

wl_data = create_shifted_df(wl_data2, 72, 72+36, 'H', 'mean', ref, True)


### Precip
# p_stn_id = [p['station_id'] for p in p_sites][0]
# p_data1 = tethys1.get_results(p_dataset['dataset_id'], p_stn_id, remove_height=True)
# ref = p_sites[0]['ref']
# p_data1 = p_data1.to_dataframe()['gage_height']
# p_data2 = p_data1.resample('H').mean().interpolate('pchip', limit=24).dropna()

# f_hours = list(range(47, 97))
# f_hours[0] = 0

# df_list = []
# for d in f_hours:
#     n1 = p_data2.shift(d, 'H')
#     n1.name = ref + '_' + str(d)
#     df_list.append(n1)
# p_data = pd.concat(df_list, axis=1).dropna()

precip_r_dict = tethys1.get_bulk_results(p_dataset['dataset_id'], p_stn_ids, remove_height=True)

ref_list = []
df_list = []
for s, df1 in precip_r_dict.items():
    df2 = df1.to_dataframe()['precipitation'].resample('H').sum().iloc[1:-1].fillna(0)
    stn = [p for p in p_sites if p['station_id'] == s][0]
    site_name = stn['ref']

    data = create_shifted_df(df2, 10, 10+n_hours_shift, 'H', 'sum', site_name)
    df_list.append(data)
    ref_list.extend([site_name])

p_data = pd.concat(df_list, axis=1).dropna()


# t_stn_id = [p['station_id'] for p in t_sites][0]
# t_data1 = tethys1.get_results(t_dataset['dataset_id'], t_stn_id, remove_height=True)
# ref = t_sites[0]['ref']
# t_data2 = t_data1.to_dataframe()['temperature']
# # f_data1 = df1.resample('D').max()
# t_data3 = t_data2.resample('H').mean().interpolate('pchip', limit=48).dropna()

# f_hours = list(range(14, n_hours_shift+1))
# # f_hours[0] = 0

# df_list = []
# for d in f_hours:
#     n1 = t_data3.shift(d, 'H')
#     n1.name = ref + '_' + str(d)
#     df_list.append(n1)
# t_data = pd.concat(df_list, axis=1).dropna()

#####################################################
#### 66401 streamflow

### Houly model

### Prepare data
label_name = '66401_0'
actual = f_data[label_name].loc[train_date_cut_off:]
actual.name = 'Actual Flow'

# data1 = pd.concat([f_data, p_data, t_data], axis=1).dropna()
data1 = pd.concat([f_data, p_data], axis=1).dropna()
# data1 = pd.concat([f_data[[label_name]], p_data], axis=1).dropna()

train_features_df = data1.loc[:train_date_cut_off].drop(label_name, axis = 1)
train_labels = np.array(data1.loc[:train_date_cut_off, label_name])
train_features = np.array(train_features_df)

test_features_df = data1.loc[train_date_cut_off:].drop(label_name, axis = 1)

test_features = np.array(test_features_df)
test_labels = np.array(actual)

# print(train_labels)
# print(train_labels.shape)
# print("")
# print(train_features)
# print(train_features.shape)

## Train model
# rf = GradientBoostingRegressor(n_estimators = 100)
# rf.fit(train_features, train_labels)

rf = HistGradientBoostingRegressor(loss='least_squares', max_iter=100, learning_rate=0.1)
rf.fit(train_features, train_labels)

# rf = RandomForestRegressor(n_estimators = 200, n_jobs=4)
# rf.fit(train_features, train_labels)

## Make the predictions and combine with the actuals
predictions1 = rf.predict(test_features)
predict1 = pd.Series(predictions1, index=test_features_df.index, name='GB Predicted Flow (m^3/s)')

combo1 = pd.merge(actual.reset_index(), predict1.reset_index(), how='left').set_index('time')

print(combo1)

### Process results
max_index = argrelextrema(test_labels, np.greater, order=12)[0]

upper_index = np.where(test_labels > np.percentile(test_labels, 80))[0]

test_labels_index = max_index[np.in1d(max_index, upper_index)]

max_data = combo1.iloc[test_labels_index]

print(max_data)

## Estimate accuracy/errors
p1 = max_data.iloc[:, 1]
a1 = max_data.iloc[:, 0]

errors = abs(p1 - a1)
bias_errors = (p1 - a1)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'm3/s.')
print('Mean Error (Bias):', round(np.mean(bias_errors), 2), 'm3/s.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / a1)
#
# Calculate and display accuracy
accuracy = np.mean(mape)
print('MANE:', round(accuracy, 2), '%.')

bias1 = np.mean(100 * (bias_errors / a1))
print('MNE:', round(bias1, 2), '%.')

bias2 = 100 * np.mean(bias_errors)/np.mean(a1)
print('NME:', round(bias2, 2), '%.')

# Get numerical feature importances -- Must be run without the Hist
importances = list(rf.feature_importances_)

# # List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(train_features_df.columns, importances)]

# # Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# # Print out the feature and importances
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))


## Plotting
ax = combo1.plot(lw=2)
max_data1 = max_data.reset_index().rename(columns={'time': 'Date', 'Actual Flow': 'Flow (m^3/s)'})
max_data1.plot.scatter('Date', 'Flow (m^3/s)', ax=ax, fontsize=15, lw=3)
# plt.show()

max_data2 = max_data1.sort_values('Flow (m^3/s)')
ax = max_data2.set_index('Flow (m^3/s)', drop=False)['Flow (m^3/s)'].plot.line(color='red', lw=2)
max_data2.plot.scatter('Flow (m^3/s)', 'GB Predicted Flow (m^3/s)', ax=ax, fontsize=15, lw=2)
# plt.show()

# print(max_data2)
max_data2 = max_data1.sort_values('Flow (m^3/s)').drop('Date', axis=1)
max_data2 = np.log(max_data2)
ax = max_data2.set_index('Flow (m^3/s)', drop=False)['Flow (m^3/s)'].plot.line(color='red', lw=2)
max_data2.plot.scatter('Flow (m^3/s)', 'GB Predicted Flow (m^3/s)', ax=ax, fontsize=15, lw=2)


##################################
### Save the model

labels = np.array(data1[label_name])
features = np.array(data1.drop(label_name, axis = 1))

rf = HistGradientBoostingRegressor(max_iter = 100)
rf.fit(features, labels)

# pkl1 = pickle.dumps(rf)

with open(os.path.join(base_path, model_file1), 'wb') as f:
    pickle.dump(rf, f)


# with open(os.path.join(base_dir, model_file), 'rb') as f:
#     rff = pickle.load(f)




