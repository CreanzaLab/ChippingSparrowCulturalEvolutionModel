import numpy as np

def bin_weight(bins, unbinned_data, bin_count_importance=1,
               bin_size_importance=1, ignore_ones=True):
    '''
    for a set of bins, calculate a weight, where more
    equal bin sizes and more equal amounts of binned
    data produce lower weights
    '''

    # since we want singletons to have a separate category, 1's are ignored
    if ignore_ones:
        unbinned_data = unbinned_data[unbinned_data != 1]
        bins = np.append(min(unbinned_data), bins)

    # add max value as right-most bin
    bins = np.append(bins, max(unbinned_data) + 1)

    bin_counts = np.unique(np.digitize(unbinned_data, bins),
                           return_counts=True)

    # accounting for bins without values: (making sure counts of 0 are
    # included)
    bin_counts = [bin_counts[1][bin_counts[0] == b] if (b in bin_counts[0])
                 else 0
                 for b in range(1, len(bins))]

    ## each bin-edge determines the size of the two adjacent bins
    # bin count weight = variance contributed by counts from adjacent bins
    bin_count_weights = (bin_counts[1:] + bin_counts[:-1]) / sum(bin_counts)
    bin_count_weights = (bin_count_weights - sum(bin_count_weights) / len(
        bin_count_weights)) ** 2

    # bin size weight = variance contributed by bin sizes of adjacent bins
    bin_size_weights = (bins[1:] - bins[:-1]) / (bins[-1] - bins[0])
    bin_size_weights = (bin_size_weights - sum(bin_size_weights) / len(
        bin_size_weights)) ** 2
    bin_size_weights = (bin_size_weights[1:] + bin_size_weights[:-1]) / 2

    weight = np.sum(bin_size_weights) ** 0.5 * bin_size_importance +\
             np.sum(bin_count_weights) ** 0.5 * bin_count_importance

    return weight

# fxn ends

# calculations:

# fixed variables
numBinsNonSingleton = 6
iterations = 1000000
np.random.seed(45)

# data to be used
counts_lifespans = [44,  0,  5,  1,  2,  0,  0,  1,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0, 0,  1,  2,  1,  0,  0,  0,  1,  1,  0,
                    1,  0,  0,  2,  0,  1,  0,  0, 0,  0,  0,  1,  0,  0,
                    1,  1,  0,  2,  2,  1,  2,  0,  0,  3,  1,  0, 4,  3,
                    2,  4,  4,  2,  2,  3,  1,  0,  4,  4,  1,  1]

counts_numSyllables = [40,  7,  8,  4,  7, 10,  1,  2,  2,  0,  2,  6,  4,
                       0,  1,  2,  2,  1, 2,  1,  1,  0,  1,  0,  1,  0,  3,
                       0,  1,  0,  0,  0,  0,  0,  2,  0, 0,  1]

for counts in [counts_lifespans, counts_numSyllables]:
    # transform counts into the data itself
    test_data = np.repeat(np.arange(1, 1 + len(counts)), counts)

    # create initial bins (set by fixed variables)
    bins = np.ceil(np.histogram_bin_edges(test_data[test_data != 1],
                                          numBinsNonSingleton))[1:-1]

    # run iterations
    for i in range(iterations):
        # in each iteration, a weight is calculated, along with each bin's contribution to that weight.
        # These weights are used to 'jitter' the bin edges accordingly, and a new weight is calculated
        # If the new weight is lower, the jittered bins are accepted.

        old_weight = bin_weight(bins, test_data)

        max_decrease = (np.append(min(test_data[test_data != 1]), bins[:-1]) -
                        bins) + 1
        max_increase = (np.append(bins[1:], len(counts) + 1) - bins)

        # the max and min changes to bin edges can cause bin edges to cross, so this should be avoided
        change_to_bins = np.zeros_like(bins)
        while (change_to_bins == np.zeros_like(bins)).all() or\
                (bins + change_to_bins != np.sort(bins + change_to_bins)).any():
            change_to_bins = np.array([np.random.randint(dec, np.ceil(inc))
                                       for dec, inc
                                       in zip(max_decrease, max_increase)])

        new_weight = bin_weight(bins + change_to_bins, test_data)

        if new_weight < old_weight:
            bins = bins + change_to_bins

    ###

    print("final weight: ", bin_weight(bins, test_data))

    bins = np.append([1, min(test_data) + 1], np.append(bins, max(test_data) + 1))

    print("bins: ", bins)

    print("bin sizes: ", bins[1:] - bins[:-1])

    print("counts: ", np.unique(np.digitize(test_data, bins),
                                return_counts=True)[1])

    print("hist: ", np.histogram(test_data, bins))


