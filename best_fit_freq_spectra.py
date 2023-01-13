from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import numpy as np
import os

from find_best_models import find_best_models
from collapse_data import collapse_array
import combine_recording_data as data_fns

c = ['k',
     '#5a7d7c',
     '#5fad56',
     '#f2c14e']

path_to_model = 'C:/Users/abiga\Box ' \
                'Sync\Abigail_Nicole\ChippiesSyllableModel' \
                '/RealYearlySamplingFreq/DispersalDist11/'

save = True
save_path = path_to_model
file_name = 'best_fit_models'

collapse = True
if collapse:
    collapse_folder = 'Collapsed_optimizedBins/'
else:
    collapse_folder = ''

"""
Set best fit models you want to visualize
"""
#load in fisher exact p-values for all models
best_neutral = find_best_models(path_to_model, 'neutral')
best_conformity = find_best_models(path_to_model, 'conformity')
best_directional = find_best_models(path_to_model, 'directional')

# learning type: [dispersal fraction, copy error rate]
best_fit_models = {
    'neutral': best_neutral,
    'conformity': best_conformity,
    'directional': best_directional
}
print(best_fit_models)

"""
Load in real song data
"""

combined_table = data_fns.load_recording_data()
# get total number of recordings for each syllable cluster
summary_table = data_fns.create_summary_table(combined_table)

"""
Load in model data
"""

base_name = "1000iters_500dim_500initSylls_40mortRate_"
model_lifetimes = {}
model_counts = {}

for model, details in best_fit_models.items():
    key_model = model + '_' + \
                details[1] + "err_" + \
                base_name + \
                details[0] + 'dispRate'
    print(key_model)
    try:
        os.chdir(path_to_model + key_model)
        model_lifetimes[key_model] = \
            pd.read_csv('sampled_lifetimes.csv',
            dtype='int32', delimiter=',',
            header=None).to_numpy().transpose()[0]
        model_counts[key_model] = \
        pd.read_csv('sampled_birds_with_syllables.csv',
                    dtype='int32', delimiter=',',
                    header=None).to_numpy().transpose()[0]
    except:
        pass

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

for key in model_lifetimes:
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

    # fishers test on counts
    observed_c = lifespans['counts'].fillna(0).to_numpy(dtype=int)
    expected_c = np.pad(bin_count_syll_types, (0, len(observed_c)-len(
        bin_count_syll_types)), 'constant')

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
plt.xlabel('Syllable Lifespan (years)')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                'Hist_LifespansVsPercentTypes/' + collapse_folder +
                file_name +
                '.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
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

for key in model_counts:
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
plt.xlabel('Number of Birds')

plt.tight_layout()
if save:
    plt.savefig(save_path +
                'Hist_NumRecVsPercentTypes/' + collapse_folder +
                file_name + '.pdf',
                type='pdf', bbox_inches='tight',
                transparent=True)
plt.show()
plt.close()