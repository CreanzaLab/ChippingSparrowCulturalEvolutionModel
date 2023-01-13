from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import numpy as np
import os
from scipy.stats import chisquare, anderson_ksamp

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
r_stats = importr('stats')

from collapse_data import collapse_array
import combine_recording_data as data_fns


"""
Set for plotting
"""
c_all = ['#54278f',
         '#5a7d7c',
         '#5fad56',
         '#f2c14e']

c_neutral = ['#54278f',
             '#acbebd',
             '#7a9796',
             '#5a7d7c',
             '#486463',
             '#364b4a']

c_conformity = ['#54278f',
                '#afd6aa',
                '#7ebd77',
                '#5fad56',
                '#4c8a44',
                '#396733']

c_directional = ['#54278f',
                 '#f8e0a6',
                 '#f4cd71',
                 '#f2c14e',
                 '#c19a3e',
                 '#91732e']


save = True
get_models_w = 'conformity'
file_name_base = get_models_w + '_degError_'

if get_models_w == 'neutral':
    c = c_neutral
elif get_models_w == 'directional':
    c = c_directional
elif get_models_w == 'conformity':
    c = c_conformity


dispersals = ['0dispRate',
              '0.1dispRate',
              '0.2dispRate',
              '0.3dispRate',
              '0.4dispRate',
              '0.5dispRate']
# dispRate = '0.1dispRate'
# file_name = 'neutral_degError_' + dispRate

load_errors = [0.0001, 0.001, 0.01, 0.1, 1.0]

collapse = True
if collapse:
    collapse_folder = 'Collapsed_optimizedBins/'
else:
    collapse_folder = ''

save_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesSyllableModel" \
            "/RealYearlySamplingFreq/DispersalDist11"

"""
Load in real song data
"""

combined_table = data_fns.load_recording_data()

"""
downsample by latitude and longitude
"""
# combined_table = combined_table.groupby(
#     ['Latitude', 'Longitude']).apply(
#     lambda x: x.sample(1, random_state=42)).reset_index(drop=True)


"""
Load in model data
"""
lifespan_fisher_df = pd.DataFrame(columns=['disp', 'error', 'pval', 'type'])
numRec_fisher_df = pd.DataFrame(columns=['disp', 'error', 'pval', 'type'])
for dispRate in dispersals:
    file_name = file_name_base + dispRate

    path_to_model = 'C:/Users/abiga\Box ' \
                    'Sync\Abigail_Nicole\ChippiesSyllableModel' \
                    '/RealYearlySamplingFreq/DispersalDist11/'

    model_details = "1000iters_500dim_500initSylls_40mortRate_" + dispRate

    model_lifetimes = {}
    model_counts = {}

    for i in ['neutral_', 'conformity_', 'directional_']:
        for j in load_errors:
            key_ij = i + str(j) + "err_" + model_details
            try:
                os.chdir(path_to_model + key_ij)
                model_lifetimes[key_ij] = pd.read_csv('sampled_lifetimes.csv',
                                                      dtype='int32', delimiter=',',
                                                      header=None).to_numpy().transpose()[
                    0]
                model_counts[key_ij] = \
                pd.read_csv('sampled_birds_with_syllables.csv',
                            dtype='int32', delimiter=',',
                            header=None).to_numpy().transpose()[0]
            except:
                pass


    """
    Make an output table for supplement that give the total number of recordings 
    in each syllable cluster, the earliest and latest observation of such syllable 
    and number of such syllable recorded in the east, west, mid, south. 
    """

    # get total number of recordings for each syllable cluster
    summary_table = data_fns.create_summary_table(combined_table)

    # summary_table.to_csv('C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesProject\StatsOfFinalData_withReChipperReExported'
    #                      '/SyllableAnalysis/SyllableClusterSummaryTable.csv')


    """
    Plot 3: long-lived lifespans
    """
    ######PLOT 3: common long-lived lifespans
    # use this information to create a histogram of the lifespan of the syllable clusters (with hues for quartiles of
    # most to least prevalent syllables)
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure(3)

    summary_table_yr, lifespan_quantile = data_fns.append_lifespans(summary_table)

    print('plot 3 data')

    lifespans = lifespan_quantile[:].sum(axis='columns')
    lifespans = lifespans.to_frame()
    lifespans.columns = ['counts']

    if collapse:
        lifespans['counts'] = collapse_array(lifespans['counts'].to_numpy(),
                                             data='lifespans').tolist()
        lifespans = lifespans[lifespans.counts != -1]

    lifespans['PercentOfTypes'] = lifespans['counts']/len(summary_table_yr)

    plot_3_df = pd.DataFrame(columns=['model', 'overlap', 'fisher pval',
                                      'fisher exact', 'chisq', 'ksamp'])
    for key in model_lifetimes:
        if str(get_models_w) in key:
            sample_lifetimes = model_lifetimes[key]

            #add model data
            num_syll_types = np.sum(sample_lifetimes > 0)
            bin_count_syll_types = np.bincount(sample_lifetimes)

            if collapse:
                bin_count_syll_types = collapse_array(bin_count_syll_types[1:],
                                                      data='lifespans')
                bin_count_syll_types = bin_count_syll_types[bin_count_syll_types
                                                            != -1]
            else:
                bin_count_syll_types = bin_count_syll_types[1:]

            y = bin_count_syll_types / num_syll_types

            print(key)

            # overlap of percent of types
            observed = lifespans['PercentOfTypes'].to_numpy()
            expected = np.pad(y, (0, len(observed)-len(y)), 'constant')
            print('overlap ', np.sum(np.minimum(observed, expected)))

            # fishers test on counts
            observed_c = lifespans['counts'].fillna(0).to_numpy(dtype=int)
            expected_c = np.pad(bin_count_syll_types, (0, len(observed_c)-len(
                bin_count_syll_types)), 'constant')

            print('size of contingency table', np.shape(np.column_stack((
                observed_c, expected_c))))

            try:
                fisher_e = 'exact'
                fisher_p = r_stats.fisher_test(np.column_stack((observed_c,
                                                                expected_c)),
                                               workspace=20000000)
            except rpy2.rinterface.RRuntimeError:
                fisher_e = 'simulated'
                fisher_p = r_stats.fisher_test(np.column_stack((observed_c,
                                                                expected_c)),
                                               workspace=20000000,
                                               simulate_p_value=True,
                                               B=1000000)

            chisq = chisquare(observed_c, expected_c)
            ksamp = anderson_ksamp([observed_c, expected_c])

            print('fisher pval: {}'.format(fisher_p[0][0]))
            print('chisq', chisq)
            print('ksamp', ksamp)

            plot_3_df = plot_3_df.append({'model': key,
                                          'overlap': str(np.sum(np.minimum(
                                             observed, expected))),
                                          'fisher pval': str(fisher_p[0][0]),
                                          'fisher exact': str(fisher_e),
                                          'chisq': str(chisq),
                                          'ksamp': str(ksamp)},
                                         ignore_index=True)
            lifespan_fisher_df = lifespan_fisher_df.append({
                'disp': dispRate.split('d')[0],
                'error': key.split('_')[1].split('e')[0],
                'pval': fisher_p[0][0],
                'type': str(fisher_e)},
                ignore_index=True)

            lifespans[key] = pd.Series(y, index=lifespans.index[:len(y)])

    # plot and save figure
    ax = lifespans.plot(kind='bar',
                        grid=None,
                        use_index=True,
                        y=lifespans.columns[1:],
                        width=0.9,
                        fontsize=10,
                        linewidth=0,
                        rot=0,
                        color=c)

    plt.ylabel('Percent of Syllable Types')

    plt.tight_layout()
    if save:
        plt.savefig(save_path +
                    "/Hist_LifespansVsPercentTypes/" + collapse_folder +
                    file_name +
                    '.pdf',
                    type='pdf', bbox_inches='tight',
                    transparent=True)
        plot_3_df.to_csv(save_path + "/Hist_LifespansVsPercentTypes/" +
                         collapse_folder + file_name + '.csv', index=False)
    plt.show()
    plt.close()

    """
    Plot 4: bird frequency spectrum
    """

    ####PLOT 4: number of recordings vs. percent syllable types
    # now make histogram of number of syllable types vs number of birds (aka number of recordings_ with each type).
    my_dpi = 96
    sns.set(style='white')
    sns.set_style('ticks')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure(4)

    # get how many syllable types (counts) were
    # recorded X number of times (NumberOfRecordings)
    numSyllablesWithNumRecordings = summary_table.groupby('NumberOfRecordings').size().reset_index(
        name='counts').reset_index(drop=True)
    numSyllablesWithNumRecordings.set_index('NumberOfRecordings', inplace=True)
    # add missing indexes filling in counts w/ NaN and replace with 0's
    new_index2 = list(range(min(numSyllablesWithNumRecordings.index), max(numSyllablesWithNumRecordings.index)+1))
    numSyllablesWithNumRecordings = numSyllablesWithNumRecordings.reindex(
        new_index2).fillna(0)

    if collapse:
        numSyllablesWithNumRecordings['counts'] = \
            collapse_array(numSyllablesWithNumRecordings['counts'].to_numpy(),
                           data='recordings').tolist()
        numSyllablesWithNumRecordings = numSyllablesWithNumRecordings[
            numSyllablesWithNumRecordings.counts != -1]

    # divide counts (number of syllable types) by total number of syllable types
    numSyllablesWithNumRecordings['PercentOfTypes'] = numSyllablesWithNumRecordings['counts']/len(
        summary_table)

    print('plot 4 data')

    plot_4_df = pd.DataFrame(columns=['model', 'overlap', 'fisher pval',
                                      'fisher exact', 'chisq', 'ksamp'])
    for key in model_counts:
        if str(get_models_w) in key:
            sample_counts = model_counts[key]

            #add model data
            num_syll_types = np.sum(sample_counts > 0)
            bin_count_syll_types = np.bincount(sample_counts)
            # x = np.arange(len(bin_count_syll_types))[1:]

            if collapse:
                # pad to be size of real data (max has 38 recordings of same type)
                bin_count_syll_types = np.pad(bin_count_syll_types, (0, 39),
                                              'constant')
                bin_count_syll_types = collapse_array(bin_count_syll_types[1:],
                                                      data='recordings')
                bin_count_syll_types = bin_count_syll_types[bin_count_syll_types
                                                            != -1]
            else:
                bin_count_syll_types = bin_count_syll_types[1:]

            y = bin_count_syll_types / num_syll_types

            print(key)

            # overlap of percent of types
            observed = numSyllablesWithNumRecordings[
                'PercentOfTypes'].fillna(0).to_numpy()

            if len(observed) > len(y):
                expected = np.pad(y, (0, len(observed)-len(y)),
                                  'constant')
            else:
                observed = np.pad(observed, (0, len(y)-len(observed)),
                                  'constant')
                expected = y

            print('overlap ', np.sum(np.minimum(observed, expected)))

            # fishers test on counts
            observed_c = numSyllablesWithNumRecordings['counts'].fillna(
                0).to_numpy(dtype=int)

            if len(observed_c) > len(bin_count_syll_types):
                expected_c = np.pad(bin_count_syll_types,
                                    (0, len(observed_c)-len(
                                        bin_count_syll_types)), 'constant')
            else:
                observed_c = np.pad(observed_c,
                                    (0, len(bin_count_syll_types)-len(
                                        observed_c)), 'constant')
                expected_c = bin_count_syll_types

            print(np.shape(np.column_stack((observed_c, expected_c))))
            try:
                fisher_e = 'exact'
                fisher_p = r_stats.fisher_test(np.column_stack((observed_c,
                                                                expected_c)),
                                               workspace=20000000)
            except rpy2.rinterface.RRuntimeError:
                fisher_e = 'simulated'
                fisher_p = r_stats.fisher_test(np.column_stack((observed_c,
                                                                expected_c)),
                                               workspace=20000000,
                                               simulate_p_value=True,
                                               B=1000000)

            chisq = chisquare(observed_c, expected_c)
            ksamp = anderson_ksamp([observed_c, expected_c])

            print('fisher pval: {}'.format(fisher_p[0][0]))
            print('chisq', chisq)
            print('ksamp', ksamp)


            plot_4_df = plot_4_df.append({'model': key,
                                          'overlap': str(np.sum(np.minimum(
                                             observed, expected))),
                                          'fisher pval': str(fisher_p[0][0]),
                                          'fisher exact': str(fisher_e),
                                          'chisq': str(chisq),
                                          'ksamp': str(ksamp)},
                                         ignore_index=True)

            numRec_fisher_df = numRec_fisher_df.append({
                'disp': dispRate.split('d')[0],
                'error': key.split('_')[1].split('e')[0],
                'pval': fisher_p[0][0],
                'type': str(fisher_e)},
                ignore_index=True)

            # numSyllablesWithNumRecordings[key] = pd.Series(y)
            if collapse:
                numSyllablesWithNumRecordings[key] = \
                    pd.Series(y, index=numSyllablesWithNumRecordings.index)
            else:
                numSyllablesWithNumRecordings = pd.concat((
                    numSyllablesWithNumRecordings,
                    pd.Series(y, index=np.arange(1, len(y)+1)).rename(key)), axis=1)


    # df index 'NumberOfRecordings' is x-axis
    # y is columns 'PercentOfTypes_' for real data and models
    # column 'counts' is not used
    ax = numSyllablesWithNumRecordings.plot(kind='bar',
                                            use_index=True,
                                            y=numSyllablesWithNumRecordings.columns[1:],
                                            grid=None,
                                            rot=0,
                                            width=0.9,
                                            fontsize=10,
                                            linewidth=0,
                                            color=c)

    plt.ylabel('Percent of Syllable Types')

    plt.tight_layout()
    if save:
        plt.savefig(save_path +
                    "/Hist_NumRecVsPercentTypes/" + collapse_folder +
                    file_name + '.pdf',
                    type='pdf', bbox_inches='tight',
                    transparent=True)
        plot_4_df.to_csv(save_path + "/Hist_NumRecVsPercentTypes/" +
                         collapse_folder + file_name + '.csv', index=False)
    plt.show()
    plt.close()

if save:
    lifespan_fisher_df.to_csv(save_path + "/Hist_LifespansVsPercentTypes/" +
                              collapse_folder + file_name_base +
                              'lifespan_fisher_df.csv',
                              index=False)
    numRec_fisher_df.to_csv(save_path + "/Hist_NumRecVsPercentTypes/" +
                            collapse_folder + file_name_base +
                              'numRec_fisher_df.csv',
                            index=False)


sns.set_context({"figure.figsize": (7, 3.5)})
ax = sns.heatmap(lifespan_fisher_df.pivot('disp', 'error', 'pval'),
                 cmap='PuOr',
                 center=0.01,
                 vmin=0,
                 vmax=1,
                 annot=True,
                 fmt='.2G',
                 annot_kws={'size': 12})
ax.invert_yaxis()

plt.xlabel('Learning Error (%)')
plt.ylabel('Fraction of Dispersal')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                "/Hist_LifespansVsPercentTypes/" + collapse_folder +
                file_name_base + 'lifespan_fisher_df_PuOr_center_flipAx.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()

ax = sns.heatmap(numRec_fisher_df.pivot('disp', 'error', 'pval'),
                 cmap='PuOr',
                 center=0.01,
                 vmin=0,
                 vmax=1,
                 annot=True,
                 fmt='.2G',
                 annot_kws={'size': 12})
ax.invert_yaxis()

# plt.ylim(0, 0.5)
plt.xlabel('Learning Error (%)')
plt.ylabel('Fraction of Dispersal')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                "/Hist_NumRecVsPercentTypes/" + collapse_folder +
                file_name_base + 'numRec_fisher_df_PuOr_center_flipAx.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()