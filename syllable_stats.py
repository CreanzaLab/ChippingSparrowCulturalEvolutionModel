from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import numpy as np
import csv
from scipy.stats import ranksums
from matplotlib.ticker import FuncFormatter

import combine_recording_data as data_fns


"""
Save info
"""
save = True
save_path = "C:/Users/abiga\Box " \
            "Sync/Abigail_Nicole/ChippiesSongEvolution/Figures"


"""
Load in song data
"""
combined_table = data_fns.load_recording_data(song_stats=False)

"""
downsample by latitude and longitude
"""
# combined_table = combined_table.groupby(
#     ['Latitude', 'Longitude']).apply(
#     lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

"""
Make an output table for supplement that give the total number of recordings in each syllable cluster, the earliest 
and latest observation of such syllable and number of such syllable recorded in the east, west, mid, south. 
"""
# # get total number of recordings for each syllable cluster
summary_table = data_fns.create_summary_table(combined_table)

if save:
    summary_table.to_csv(save_path + '/SyllableClusterSummaryTable.csv')

"""
PLOT 3: common long-lived lifespans
use the summary table information to create a histogram of the lifespan of the 
syllable clusters (with hues for quartiles of most to least prevalent 
syllables)
"""
my_dpi = 96
sns.set(style='white')
sns.set_style('ticks')
sns.set_context({"figure.figsize": (20, 7)})

summary_table_yr, lifespan_quantile = data_fns.append_lifespans(summary_table)

# plot and save figure
ax = lifespan_quantile.plot(kind='bar', stacked=True, grid=None, width=1, fontsize=10, edgecolor='black', rot=0,
                            color=['#cbc9e2', '#9e9ac8', '#756bb1', '#54278f'])# sns.color_palette("PRGn", 10))

plt.tight_layout()

if save:
    plt.savefig(save_path + "/HistogramOfClusterLifespans_stackedQuantiles"
                + '.pdf',
                type='pdf',
                bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()

"""
PLOT 4: number of recordings vs. percent syllable types
make histogram of number of syllable types vs number of birds 
(aka number of recordings_ with each type).
"""
my_dpi = 96
sns.set(style='white')
sns.set_style('ticks')
sns.set_context({"figure.figsize": (15, 7)})

numSyllablesWithNumRecordings = summary_table.groupby('NumberOfRecordings').size().reset_index(
    name='counts').reset_index(drop=True)
numSyllablesWithNumRecordings['PercentOfTypes'] = numSyllablesWithNumRecordings['counts']/len(
    summary_table)

# above_thresh = numSyllablesWithNumRecordings[numSyllablesWithNumRecordings['NumberOfRecordings'] >= 20]
# percent_numRec = (above_thresh['NumberOfRecordings'].astype(float)*above_thresh['counts']).sum()/(
#     numSyllablesWithNumRecordings['NumberOfRecordings']*numSyllablesWithNumRecordings['counts']).sum()
# percent_syllTypes = above_thresh['PercentOfTypes'].sum()
# print('percent num rec', percent_numRec)
# print('percent syll types', percent_syllTypes)

numSyllablesWithNumRecordings.set_index('NumberOfRecordings', inplace=True)
new_index2 = list(range(min(numSyllablesWithNumRecordings.index), max(numSyllablesWithNumRecordings.index)+1))
numSyllablesWithNumRecordings = numSyllablesWithNumRecordings.reindex(new_index2)

ax = numSyllablesWithNumRecordings.plot(kind='bar', use_index=True, y=['PercentOfTypes'], stacked=True,
                                        grid=None,
                                        width=1, fontsize=10, edgecolor='black', color='#cbc9e2', rot=0)

plt.tight_layout()

if save:
    plt.savefig(save_path + "/HistogramOfNumberOfRecordingsVSPercentOfTypes"
                + '.pdf',
                type='pdf',
                bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()


"""
Look at what is common in syllables that are short or long-lived
"""

my_dpi = 96
sns.set(style='white')
sns.set_style('ticks')
sns.set_context({"figure.figsize": (15, 7)})

###PLOT 5: lifespan by syllable category
# group by lifespan and syllable category
summary_table_yr['lifespan_group'] = np.nan
summary_table_yr['lifespan_group'] = np.where(summary_table_yr['Lifespan'] == 1, 'short-lived', summary_table_yr[
    'lifespan_group'])
summary_table_yr['lifespan_group'] = np.where(summary_table_yr['Lifespan'] >= 50, 'long-lived', summary_table_yr[
    'lifespan_group'])
summary_table_yr['lifespan_group'] = summary_table_yr[
    'lifespan_group'].replace('nan', np.nan)
summary_table_yr = summary_table_yr.dropna(subset=['lifespan_group'])

numSyllTypesWithLifespan = summary_table_yr.groupby(['lifespan_group', 'Category']).size().reset_index(
    name='counts').reset_index(drop=True)

order = ['updown', 'buzz', 'downup', 'sweep', 'double', 'complex']
numSyllTypesWithLifespan = numSyllTypesWithLifespan.pivot(columns='lifespan_group', index='Category').loc[
    order]
numSyllTypesWithLifespan.columns = numSyllTypesWithLifespan.columns.droplevel()
numSyllTypesWithLifespan = numSyllTypesWithLifespan.reindex_axis(['short-lived', 'long-lived'], axis=1)

# plot and save figure
ax = numSyllTypesWithLifespan.plot(kind='bar', stacked=False, width=0.5, grid=None, fontsize=10,
                        color=['#cbc9e2', '#54278f'], edgecolor='black', rot=0)
plt.tight_layout()

if save:
    plt.savefig(save_path + "/Longevity_SyllableCategories"
                + '.pdf',
                type='pdf',
                bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()

"""
BOX PLOTS and WILCOXON OF SONG FEATURES FOR LONGEVITY
"""

# get rid of unnecessary metadata and the song stat variables (had to do this using the original dataframe because
# earlier some of the wanted metada had been dropped previously)

combined_table_withFeatures = data_fns.load_recording_data(song_stats=True)

longevity_dict = summary_table_yr[['lifespan_group']].to_dict()['lifespan_group']
combined_table_withFeatures['longevity'] = combined_table_withFeatures['ClusterNoAdjusted'].map(longevity_dict)

song_variables = ['Mean Note Duration',
                  'Mean Note Frequency Modulation',
                  'Mean Note Frequency Trough',
                  'Mean Note Frequency Peak',
                  'Mean Inter-Syllable Silence Duration',
                  'Mean Syllable Duration',
                  'Mean Syllable Frequency Modulation',
                  'Mean Syllable Frequency Trough',
                  'Mean Syllable Frequency Peak',
                  'Duration of Song Bout',
                  'Mean Stereotypy of Repeated Syllables',
                  'Number of Notes per Syllable',
                  'Syllable Rate',
                  'Total Number of Syllables',
                  'Standard Deviation of Note Duration',
                  'Standard Deviation of Note Frequency Modulation']

adjust = 4
log_var = {4: 'ms', 8: 'ms', 9: 'ms', 15: 'number', 17: 'number', 18: 'ms'}
log_convert_var = {5: 'kHz', 6: 'kHz', 7: 'kHz', 10: 'kHz', 11: 'kHz',
                   12: 'kHz', 13: 'seconds'}
log_convert_inverse_var = {16: 'number/second'}
no_log = {14: '%'}
no_log_convert = {19: 'kHz'}

# take e^x for y-axis
for key, value in log_var.items():
    fig = plt.figure(figsize=(7, 11))
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    ax = sns.boxplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                     color='None',
                     fliersize=0, width=0.5,
                     linewidth=2, order=['short-lived', 'long-lived'])
    ax = sns.stripplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                       order=['short-lived', 'long-lived'],
                       palette=['#cbc9e2', '#54278f'],
                       size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(song_variables[key-adjust] + ' (' + value + ')', fontsize=30)
    print(song_variables[key-adjust], combined_table_withFeatures.columns[key])
    ax.set_xlabel('')
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
    plt.setp(ax.spines.values(), linewidth=2)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "%.1f" % (np.exp(x))))

    if save:
        plt.savefig(save_path + "/SyllableBoxPlots/"
                    + combined_table_withFeatures.columns[key]
                    + '_noLogAxis_largerFont'
                    + '.pdf',
                    type='pdf',
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    plt.show()

# take e^x for each variable and also convert from Hz to kHz or ms to seconds
for key, value in log_convert_var.items():
    fig = plt.figure(figsize=(7, 11))
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    ax = sns.boxplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                     color='None',
                     fliersize=0, width=0.5,
                     linewidth=2, order=['short-lived', 'long-lived'])
    ax = sns.stripplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                       order=['short-lived', 'long-lived'],
                       palette=['#cbc9e2', '#54278f'],
                       size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(song_variables[key-adjust] + ' (' + value + ')', fontsize=30)
    print(song_variables[key-adjust], combined_table_withFeatures.columns[key])
    ax.set_xlabel('')
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
    plt.setp(ax.spines.values(), linewidth=2)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "%.1f" % (np.exp(x)/1000)))

    if save:
        plt.savefig(save_path + "/SyllableBoxPlots/"
                    + combined_table_withFeatures.columns[key]
                    + '_noLogAxis_largerFont'
                    + '.pdf',
                    type='pdf',
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    plt.show()

# take e^x for each variable and convert from 1/ms to 1/seconds
for key, value in log_convert_inverse_var.items():
    fig = plt.figure(figsize=(7, 11))
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    ax = sns.boxplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                     color='None',
                     fliersize=0, width=0.5,
                     linewidth=2, order=['short-lived', 'long-lived'])
    ax = sns.stripplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                       order=['short-lived', 'long-lived'],
                       palette=['#cbc9e2', '#54278f'],
                       size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(song_variables[key-adjust] + ' (' + value + ')', fontsize=30)
    print(song_variables[key-adjust], combined_table_withFeatures.columns[key])
    ax.set_xlabel('')
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
    plt.setp(ax.spines.values(), linewidth=2)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "%.1f" % (np.exp(x)*1000)))

    if save:
        plt.savefig(save_path + "/SyllableBoxPlots/"
                    + combined_table_withFeatures.columns[key]
                    + '_noLogAxis_largerFont'
                    + '.pdf',
                    type='pdf',
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    plt.show()

# are not log(value) so no need to take exponential and no conversion
for key, value in no_log.items():
    fig = plt.figure(figsize=(7, 11))
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    ax = sns.boxplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                     color='None',
                     fliersize=0, width=0.5,
                     linewidth=2, order=['short-lived', 'long-lived'])
    ax = sns.stripplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                       order=['short-lived', 'long-lived'],
                       palette=['#cbc9e2', '#54278f'],
                       size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(song_variables[key-adjust] + ' (' + value + ')', fontsize=30)
    print(song_variables[key-adjust], combined_table_withFeatures.columns[key])
    ax.set_xlabel('')
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
    plt.setp(ax.spines.values(), linewidth=2)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "%.1f" % x))

    if save:
        plt.savefig(save_path + "/SyllableBoxPlots/"
                    + combined_table_withFeatures.columns[key]
                    + '_noLogAxis_largerFont'
                    + '.pdf',
                    type='pdf',
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    plt.show()

# are not log(value) so no need to take exponential, convert from Hz to kHz
for key, value in no_log_convert.items():
    fig = plt.figure(figsize=(7, 11))
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    ax = sns.boxplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                     color='None',
                     fliersize=0, width=0.5,
                     linewidth=2, order=['short-lived', 'long-lived'])
    ax = sns.stripplot(x='longevity', y=combined_table_withFeatures.columns[key], data=combined_table_withFeatures[['longevity', combined_table_withFeatures.columns[key]]],
                       order=['short-lived', 'long-lived'],
                       palette=['#cbc9e2', '#54278f'],
                       size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(song_variables[key-adjust] + ' (' + value + ')', fontsize=30)
    print(song_variables[key-adjust], combined_table_withFeatures.columns[key])
    ax.set_xlabel('')
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
    plt.setp(ax.spines.values(), linewidth=2)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "%.1f" % (x/1000)))

    if save:
        plt.savefig(save_path + "/SyllableBoxPlots/"
                    + combined_table_withFeatures.columns[key]
                    + '_noLogAxis_largerFont'
                    + '.pdf',
                    type='pdf',
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    plt.show()

""""
Wilcoxon Ranksums
"""
if save:
    #16 song variables
    with open(save_path +
              '/SyllableBoxPlots/longevitySongFeatures_WilcoxonRanksums.csv',
              'w', newline='') as file:
        filewriter = csv.writer(file, delimiter=',')
        filewriter.writerow(['Song Feature',
                             'Short-Lived vs Long-Lived p-value'])
        for sv in combined_table_withFeatures.columns[4:20]:
            s = combined_table_withFeatures.loc[combined_table_withFeatures['longevity'] == 'short-lived', sv]
            l = combined_table_withFeatures.loc[combined_table_withFeatures['longevity'] == 'long-lived', sv]
            filewriter.writerow([sv, ranksums(s, l)[1]])

    #longitude
    metadata = ['Longitude']
    with open(save_path +
              '/SyllableBoxPlots/longevityLongitude_WilcoxonRanksums.csv',
              'w', newline='') as file:
        filewriter = csv.writer(file, delimiter=',')
        filewriter.writerow(['Metadata',
                             'Short-Lived vs Long-Lived p-value'])
        for sv in metadata:
            s = combined_table_withFeatures.loc[combined_table_withFeatures['longevity'] == 'short-lived', sv]
            l = combined_table_withFeatures.loc[combined_table_withFeatures['longevity'] == 'long-lived', sv]

            filewriter.writerow([sv, ranksums(s, l)[1]])





