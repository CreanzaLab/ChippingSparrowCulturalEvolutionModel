import pandas as pd


def load_recording_data(song_stats=False):
    """
    Load in song data
    """

    # song meta data
    data_path1 = \
        'citizen_science_data' \
        '/AnimalBehaviour_SupplementalDataTable2_addedMid_addedYears.csv'
    log_song_data = pd.read_csv(data_path1, header=0, index_col=None)
    log_song_data_unique = log_song_data.loc[log_song_data[
        'ComparedStatus'].isin(['unique', 'use'])].copy().reset_index(drop=True)

    # get rid of unnecessary metadata and the song stat variables
    if song_stats:
        col_to_skip = ['FromDatabase', 'ComparedStatus', 'RecordingDay',
                       'RecordingMonth', 'RecordingTime', 'Region']
    else:
        col_to_skip = ['FromDatabase', 'ComparedStatus', 'RecordingDay',
                       'RecordingMonth', 'RecordingTime', 'Region'] + \
                      list(log_song_data_unique.columns[10:26].values)
    song_info = log_song_data_unique.drop(col_to_skip, axis=1)

    """
    Load in syllable cluster data
    """
    # syllable clusters
    data_path2 = 'citizen_science_data/SyllableClusters_820UniqueUse.csv'
    cluster_data = pd.read_csv(data_path2, header=0, index_col=None)

    col_to_skip2 = ['SyllableNumber', 'ClusterNo']
    cluster_data = cluster_data.drop(col_to_skip2, axis=1)
    cluster_data['ClusterNoAdjusted'] = cluster_data[
        'ClusterNoAdjusted'].astype(int)

    """
    combine tables using CatalogNo
    """
    combined_table = song_info.merge(cluster_data, how='inner', on='CatalogNo')
    combined_table = combined_table.drop_duplicates(['CatalogNo',
                                                     'ClusterNoAdjusted'],
                                                    keep='first')
    combined_table = combined_table.drop(['FileName'], axis=1)

    return combined_table


def create_summary_table(combined_table):
    # get total number of recordings for each syllable cluster
    cluster_num_rec = combined_table.groupby(
        'ClusterNoAdjusted').size().reset_index(
        name='NumberOfRecordings').sort_values('ClusterNoAdjusted').set_index(
        'ClusterNoAdjusted')

    cluster_category = combined_table.drop_duplicates('ClusterNoAdjusted')[
        ['ClusterNoAdjusted',
         'Category']].sort_values(
        'ClusterNoAdjusted').set_index('ClusterNoAdjusted')

    earliest_latest_rec = combined_table.assign(
        EarliestYear=combined_table['RecordingYear'].abs(),
        LatestYear=combined_table[
            'RecordingYear'].abs()).groupby('ClusterNoAdjusted').agg(
        {'EarliestYear': 'min', 'LatestYear': 'max'})
    earliest_latest_rec = earliest_latest_rec.fillna(0).astype(int)

    summary_table = pd.concat(
        [cluster_num_rec, earliest_latest_rec, cluster_category], axis=1)
    summary_table = summary_table.reindex_axis(
        ['NumberOfRecordings', 'EarliestYear', 'LatestYear', 'Category'],
        axis=1)

    return summary_table


def append_lifespans(summary_table):
    # remove recordings without a recording year (this was 4 songs)
    summary_table_yr = summary_table[summary_table['EarliestYear'] != 0]
    # calculate lifespan
    summary_table_yr = summary_table_yr.assign(
        Lifespan=(summary_table_yr['LatestYear'] - summary_table_yr[
            'EarliestYear'] + 1))

    print(pd.qcut(summary_table_yr['NumberOfRecordings'], q=5,
                  duplicates='drop'))
    summary_table_yr['quantile'] = pd.qcut(
        summary_table_yr['NumberOfRecordings'], 5, duplicates='drop',
        labels=False)

    lifespan_quantile = summary_table_yr.groupby(
        ['quantile', 'Lifespan']).size().reset_index(name='count').pivot(
        columns='quantile', index='Lifespan')

    new_index = list(
        range(min(lifespan_quantile.index), max(lifespan_quantile.index) + 1))
    lifespan_quantile = lifespan_quantile.reindex(new_index)

    return summary_table_yr, lifespan_quantile
