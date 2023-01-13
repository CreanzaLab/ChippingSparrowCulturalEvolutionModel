import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns; sns.set()
from scipy.stats import truncnorm


def initiate(min_type, max_type, min_rate, max_rate, dim1, vector_size):
    np.random.seed(49)
    # populate each element (territory) with a random bird (syllable syll)
    # low (inclusive), high(exclusive), discrete uniform distribution
    bird_matrix = np.random.randint(min_type, max_type, size=[dim1, dim1],
                                    dtype='int')
    # assign a random syllable rate to each bird (territory)
    # low (inclusive), high(exclusive), continuous uniform distribution
    # rate_matrix = np.random.uniform(min_rate, max_rate, size=[dim1, dim1])
    mu, sigma = np.mean([min_rate, max_rate]), 5
    rate_matrix = truncnorm.rvs((min_rate - mu)/sigma,
                                (max_rate - mu)/sigma,
                                loc=mu, scale=sigma,
                                size=[dim1, dim1],)

    # initiate vectors for tracking info about entire population
    current_bps = np.zeros(vector_size, dtype='int')  # no. birds per syllable at one time step
    actual_lifetimes = np.zeros(vector_size, dtype='int')  # no. yrs syllable has existed
    unique_bps = np.zeros(vector_size, dtype='int')  # no. of individual birds that have ever had that syllable syll


    # initiate vectors for tracking info about only the sampled population
    sample_bps = np.zeros(vector_size, dtype='int')
    sample_unique_bps = np.zeros(vector_size, dtype='int')
    first_sampled = np.zeros(vector_size, dtype='int')  # first iter syllable was sampled
    last_sampled = np.zeros(vector_size, dtype='int')  # last iter syllable was sampled + 1 (so can subtract to get lifespans)

    return bird_matrix, rate_matrix, current_bps, actual_lifetimes, \
        unique_bps, sample_bps, sample_unique_bps, first_sampled, \
        last_sampled


def count_type(territories, vector_size):
    unique, counts = np.unique(territories, return_counts=True)
    counts_array = np.zeros(vector_size, dtype='int')
    for u, c in zip(unique, counts):
        counts_array[u] = c

    return counts_array


def locate_dead_birds(ordered_pairs, num_loc):
    np.random.shuffle(ordered_pairs)
    loc_deaths = ordered_pairs[:num_loc]

    return loc_deaths


def get_nearby_sylls(matrix_1, row, col, d=1):
    # does not wrap the boundaries (boundaries are real irl)

    # store value of dead bird's syllable
    dead_bird_syll = int(matrix_1[row, col])

    # to determine indices for surrounding neighbor squares
    row_start = row - d
    row_end = row + d + 1
    col_start = col - d
    col_end = col + d + 1

    # deal with edge/corner cases
    if row_start < 0:
        row_start = 0
    if row_end > len(matrix_1):
        row_end = len(matrix_1)
    if col_start < 0:
        col_start = 0
    if col_end > len(matrix_1):
        col_end = len(matrix_1)

    # get values of neighbors
    n_syll_types = matrix_1[row_start:row_end, col_start:col_end].flatten().tolist()

    # remove syll of interest from list of neighboring sylls
    n_syll_types.remove(dead_bird_syll)  # only removes first matching element

    return n_syll_types


def get_nearby_sylls_and_rates(matrix_1, matrix_2, row, col, d=1):
    # does not wrap the boundaries (boundaries are real irl)

    # store value of dead bird rate
    dead_bird_rate = float(matrix_2[row, col])

    # to determine indices for surrounding neighbor squares
    row_start = row - d
    row_end = row + d + 1
    col_start = col - d
    col_end = col + d + 1

    # deal with edge/corner cases
    if row_start < 0:
        row_start = 0
    if row_end > len(matrix_1):
        row_end = len(matrix_1)
    if col_start < 0:
        col_start = 0
    if col_end > len(matrix_1):
        col_end = len(matrix_1)

    # get values of neighbors (both syllable types and syllable rates)
    n_syll_types = matrix_1[row_start:row_end,
                            col_start:col_end].flatten().tolist()
    n_syll_rates = matrix_2[row_start:row_end,
                            col_start:col_end].flatten().tolist()

    # must remove from n_syll_rates first since you need the index from the
    # full n_syll_types list before removing dead_bird_syll
    n_syll_types.pop(n_syll_rates.index(dead_bird_rate))
    n_syll_rates.remove(dead_bird_rate)  # only removes first matching element

    return n_syll_types, n_syll_rates


def get_learned_syll(sylls_to_copy, num_sylls, rule='neutral',
                     error_rate=None, conformity_factor=1):

    if np.random.random() < error_rate:
        new_syll_type = num_sylls + 1  # re-invention is not possible, always a new syllable syll
        num_sylls += 1
    elif rule == 'neutral':  # a random nearby song
        new_syll_type = np.random.choice(sylls_to_copy)
    elif rule == 'conformity':  # most common value heard nearby, randomly chooses from ties
        nearby_uniques, nearby_counts = np.unique(sylls_to_copy,
                                                  return_counts=True)
        nearby_counts_scaled = (nearby_counts**conformity_factor).astype(int)
        new_syll_type = np.random.choice(np.repeat(nearby_uniques,
                                                   nearby_counts_scaled))
        # new_syll_type = np.random.choice(nearby_uniques,
        #                                  p=nearby_counts_scaled/np.sum(
        #                                      nearby_counts_scaled))
        # new_syll_type = np.random.choice(np.where(np.bincount(sylls_to_copy) == np.bincount(sylls_to_copy).max())[0])

    return new_syll_type, num_sylls


def get_learned_syll_and_rate(sylls_to_copy, rates_to_copy, num_sylls,
                              error_rate=None):

    # find the largest syllable rate among the neighbors
    new_syll_rate = max(rates_to_copy)
    rate_copied_idx = rates_to_copy.index(max(rates_to_copy))
    new_syll_type = sylls_to_copy[rate_copied_idx]

    if np.random.random() < error_rate:
        new_syll_type = num_sylls + 1  # re-invention is not possible, always a new syllable syll
        num_sylls += 1

    # get random error for for how well the bird learned a slower or faster
    # rate than the tutor's rate
    rate_error = np.random.uniform(-2, 0.25)
    new_syll_rate += rate_error
    if new_syll_rate > 40:
        new_syll_rate = float(40)
    elif new_syll_rate < 1:
        new_syll_rate = float(1)

    return new_syll_type, new_syll_rate, num_sylls


def sample_birds(all_territories, sampling_num):
    # get random matrix locations (x, y)
    loc_samples = np.random.randint(0, len(all_territories), size=(sampling_num, 2))  # by chance could get the same element twice
    sample_sylls = []
    for sample in loc_samples:
        # get syllable syll of the sampled bird
        sample_sylls.append(all_territories[sample[0], sample[1]])

    return sample_sylls


def adult_dispersal(dispersal_frac, d, max_try, matrix_dim):
    dispersal_num = int(dispersal_frac * (matrix_dim ** 2))

    # track who has already moved
    already_dispersed = np.zeros((matrix_dim, matrix_dim), dtype=np.bool)
    loc_samples = np.random.randint(0, matrix_dim, size=(dispersal_num, 2))

    loc_swaps = []
    for sample_1 in loc_samples:
        # check if the bird has already dispersed
        if already_dispersed[sample_1[0], sample_1[1]]:
            loc_swaps.append(None)
        else:
            # get indices of neighbors within some dispersal threshold

            row = sample_1[0]
            col = sample_1[1]

            # to determine indices for surrounding neighbor squares
            row_start = row - d
            row_end = row + d + 1
            col_start = col - d
            col_end = col + d + 1

            # deal with edge/corner cases
            if row_start < 0:
                row_start = 0
            if row_end > matrix_dim:
                row_end = matrix_dim
            if col_start < 0:
                col_start = 0
            if col_end > matrix_dim:
                col_end = matrix_dim

            # get second bird's location and check if they have already
            # dispersed, if so, find a different second bird to swap with
            second_dispersed = True
            count = 0
            while second_dispersed and count < max_try:
                count += 1
                sample_2 = (np.random.randint(row_start, row_end, 1),
                            np.random.randint(col_start, col_end, 1))
                if already_dispersed[sample_2[0], sample_2[1]]:
                    second_dispersed = True
                else:
                    loc_swaps.append(sample_2)
                    already_dispersed[sample_2[0], sample_2[1]] = True
                    second_dispersed = False

    return loc_samples, loc_swaps


def plot_type_distributions(types_matrix, t, bin_size=10):
    uniq_sylls, num_birds_w_syll = np.unique(types_matrix,
                                             return_counts=True)
    num_unique_sylls_in_matrix = len(uniq_sylls)
    bin_count_num_birds = np.bincount(num_birds_w_syll)
    count_binned = [bin_count_num_birds[n:n + bin_size] for n in
                    range(0, len(bin_count_num_birds), bin_size)]
    count_binned = [np.sum(count_binned[i]) for i in
                    range(0, len(count_binned))]

    y = count_binned.copy()
    x = np.arange(len(y))

    b = y.copy()
    a = x.copy() / np.sum(num_birds_w_syll) * bin_size * 100

    # raw bird counts
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure()

    plt.bar(x, y)
    plt.title('number unique syllables at time ' + str(t) + ': '
              + str(num_unique_sylls_in_matrix))
    plt.xlabel('no. birds singing a syllable x' + str(bin_size))
    plt.ylabel('no. of syllable types')

    plt.tight_layout()
    plt.savefig(
        "dist_bird_in_matrix_" + str(t)
        + '.pdf', type='pdf', bbox_inches='tight',
        transparent=True)
    plt.close()

    # percent of population
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure()

    plt.bar(a, b, width=0.003)
    plt.title('number unique syllables at time ' + str(t) + ': '
              + str(num_unique_sylls_in_matrix))
    plt.xlabel('percent of birds singing a syllable')
    plt.ylabel('no. of syllable types')
    # plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        "percent_bird_in_matrix_" + str(t)
        + '.pdf', type='pdf', bbox_inches='tight',
        transparent=True)
    plt.close()


def plot_rates_distributions(rates_matrix, t, bin_size=10):
    # percent of population
    my_dpi = 96
    sns.set(style='white')
    sns.set_context({"figure.figsize": (20, 7)})
    plt.figure()

    plt.hist(rates_matrix.flatten())
    plt.title('time ' + str(t))
    plt.xlabel('rate (syllables/seconds)')
    plt.ylabel('number of birds with rate')
    # plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        "syll_rates_in_matrix_" + str(t)
        + '.pdf', type='pdf', bbox_inches='tight',
        transparent=True)
    plt.close()
