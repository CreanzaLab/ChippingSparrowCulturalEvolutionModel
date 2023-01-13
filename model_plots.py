import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
import os
import sys
from glob import glob


"""
Load in counts from run
"""

home_path = 'C:/Users/abiga\Box ' \
            'Sync\Abigail_Nicole\ChippiesSyllableModel' \
            '/RealYearlySamplingFreq/Testing4_new/500DimMatrix' \
            '/ForDissertation'

folders = glob(home_path + "/*/")

for f in folders:
    print(f)

    os.chdir(f)

    birds_t0 = np.genfromtxt('bird_counts_t0.csv',
                                     dtype='int32', delimiter=',')
    birds_t1950 = np.genfromtxt('bird_counts_t1950.csv',
                                     dtype='int32', delimiter=',')
    birds_t2017 = np.genfromtxt('bird_counts_t2017.csv',
                                     dtype='int32', delimiter=',')

    print('no. nonzero t0: ', np.sum(birds_t0 > 0))
    print('no. nonzero t1950: ', np.sum(birds_t1950 > 0))
    print('no. nonzero t2017: ', np.sum(birds_t2017 > 0))

    actual_lifetimes = np.genfromtxt('actual_lifetimes_starting_1950.csv',
                                     dtype='int32', delimiter=',')
    actual_counts = np.genfromtxt('unique_birds_with_syllables.csv',
                                  dtype='int32', delimiter=',')
    sampled_lifetimes = pd.read_csv('sampled_lifetimes.csv',
                                      dtype='int32', delimiter=',',
                                    header=None).to_numpy().transpose()[0]
    print('loaded first', sampled_lifetimes.shape)
    print(sampled_lifetimes[0:10])
    sample_counts = pd.read_csv('sampled_birds_with_syllables.csv',
                                dtype='int32', delimiter=',',
                                header=None).to_numpy().transpose()[0]
    print('loaded', sample_counts.shape)

    # print('total number of syllable types: ', np.sum(actual_counts > 0))

    print('(sampled) total number of syllable types: ', np.sum(sample_counts
                                                               > 0))

    """
    Plotting results
    """

    save_plots = True

    # lifetime of syllables versus number of syllable types (for sample)
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure(4)

    num_syll_types = np.sum(sample_counts > 0)
    x = np.arange(max(sampled_lifetimes))
    y = np.bincount(sampled_lifetimes)
    print(y.shape)

    # don't include no. with 0 lifespan (will be a lot since vector was made
    # much larger than needed)
    plt.bar(x + 1, y[1:])
    plt.title('Sampled: ' + ' total number of syll types: ' + str(
        num_syll_types))
    plt.xlabel('lifespan')
    plt.ylabel('number of syllable types')
    # plt.xlim(0, 80)
    # plt.xticks(range(1, 201))
    # plt.ylim(0, 450)

    if save_plots:
        plt.tight_layout()
        plt.savefig(
            "num_sylls_with_lifespan_sampled"
            + '.pdf', type='pdf', bbox_inches='tight',
            transparent=True)
        plt.close()
    else:
        plt.show()
        plt.close()

    print('plot 1')

    # percent of syllable types being sung by some number of birds (for sample)
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure(5)

    num_syll_types = np.sum(sample_counts > 0)
    bin_count_syll_types = np.bincount(sample_counts)
    x = np.arange(len(bin_count_syll_types))
    y = bin_count_syll_types/num_syll_types

    plt.bar(x[1:], y[1:])
    plt.title('Sampled: ' + ' total number of syll types: ' + str(
        num_syll_types))
    plt.xlabel('number of birds with a syllable type')
    plt.ylabel('fraction of syllable types')
    # plt.xlim(0, 45)
    # plt.xticks(range(1, 201))
    plt.ylim(0, 1)

    if save_plots:
        plt.tight_layout()
        plt.savefig(
            "num_sylls_with_num_birds_sampled"
            + '.pdf', type='pdf', bbox_inches='tight',
            transparent=True)
        plt.close()
    else:
        plt.show()
        plt.close()

    print('plot 2')

    # percent of syllable types being sung by some number of birds
    # (full distribution from same time points as the sampling)
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure(6)

    num_syll_types = np.sum(actual_counts > 0)
    bin_count_syll_types = np.bincount(actual_counts)[1:]  # don't plot syllables that never existed

    count_binned = [bin_count_syll_types[n:n + 10] for n in range(0, len(bin_count_syll_types), 10)]
    count_binned = [np.sum(count_binned[i]) for i in range(0, len(count_binned))]

    x = np.arange(len(count_binned))
    y = count_binned / num_syll_types

    plt.bar(x, y)
    plt.title('After 1950: ' + ' total number of syll types: ' + str(
        num_syll_types))
    plt.xlabel('number of birds with a syllable type (x10)')
    plt.ylabel('fraction of syllable types')
    # plt.xlim(0, 45)
    # plt.xticks(range(1, 201))
    plt.ylim(0, 1)

    if save_plots:
        plt.tight_layout()
        plt.savefig(
            "num_sylls_with_num_birds_actual"
            + '.pdf', type='pdf', bbox_inches='tight',
            transparent=True)
        plt.close()
    else:
        plt.show()
        plt.close()


