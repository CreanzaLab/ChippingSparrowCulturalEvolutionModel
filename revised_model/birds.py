import numpy as np
from scipy.stats import truncnorm


class BirdModel:

    def __init__(self, model_type,
                 dispersal_rate, error_rate,
                 conformity_factor=2,
                 dispersal_dist=11, dim=500,
                 min_type=0, max_type=500,
                 min_rate=5, max_rate=40,
                 mortality=0.4,
                 logfile=None,
                 total_generations=1000,
                 sampled_generations=68):
        """ initialize the model for iteration """
        np.random.seed(49)
        # populate each element (territory) with a random bird (syllable syll)
        # low (inclusive), high(exclusive), discrete uniform distribution
        self.bird_matrix = np.random.randint(
            min_type, max_type,
            size=(dim, dim), dtype='int')

        # only used for directional selection
        mu = (max_rate + min_rate) / 2
        sigma = 5
        trunc = (max_rate - min_rate) / (2 * sigma)
        self.rate_matrix = truncnorm.rvs(
            -trunc, trunc,  # number of std deviations to left or right
            loc=mu, scale=sigma,
            size=(dim, dim))

        self.history = []
        self.open_territories = None
        self.generations = 0
        self.dim = dim
        self.mortality = mortality
        self.model_type = model_type
        self.error_rate = error_rate
        self.conformity_factor = conformity_factor
        self.dispersal_rate = dispersal_rate
        self.dispersal_dist = dispersal_dist
        self.logfile=logfile
        self.total_generations = total_generations
        self.generations_for_burnin = total_generations - sampled_generations
        self.sampled_generations = sampled_generations

    def death_step(self):
        """
        select a subset of individuals that die this time-step
        """
        self.open_territories = np.where(
            np.random.random((self.dim, self.dim)) < self.mortality
        )

    def fill_open(self, distance=1):
        """
        each territory must be filled by selecting syllables from adjacent birds
        """

        # get new syllables for birds that will now occupy empty territories
        new_syllables = []
        new_rates = []
        max_syllable = np.max(self.bird_matrix)
        for x, y in zip(*self.open_territories):
            # identify neighbors
            row_0 = max(0, x - distance)
            row_1 = min(self.dim, x + distance + 1)
            col_0 = max(0, y - distance)
            col_1 = min(self.dim, y + distance + 1)

            if np.random.rand() < self.error_rate:
                # make a new syllable (error)
                max_syllable += 1
                new_syllables.append(max_syllable)
                if self.model_type == 'directional':
                    neighbor_rates = self.rate_matrix[row_0:row_1, col_0:col_1].flatten().tolist()
                    neighbor_rates.remove(self.rate_matrix[x, y])
                    rate = max(neighbor_rates) + np.random.uniform(-2, 0.25)
                    rate = max(rate, 1)
                    rate = min(rate, 40)
                    new_rates.append(rate)

                continue

            # if not making a new syllable, choose a syllable from among the choices
            # start by getting syllables from the adjacent territories

            neighbor_sylls: list = self.bird_matrix[row_0:row_1, col_0:col_1].flatten().tolist()
            neighbor_sylls.remove(self.bird_matrix[x, y])

            if self.model_type == 'neutral':  # a random nearby song
                new_syllables.append(
                    np.random.choice(neighbor_sylls))

            if self.model_type == 'conformity':  # prefer common syllables
                nearby_uniques, nearby_counts = np.unique(neighbor_sylls,
                                                          return_counts=True)
                syll_conformity = (nearby_counts ** self.conformity_factor)
                syll_conformity /= np.sum(syll_conformity)
                new_syllables.append(
                    np.random.choice(nearby_uniques, p = syll_conformity))

            if self.model_type == 'directional':  # based on highest syllable trill rate
                neighbor_rates = self.rate_matrix[row_0:row_1, col_0:col_1].flatten().tolist()
                neighbor_rates.remove(self.rate_matrix[x, y])

                # find the maximum rate
                rate = max(neighbor_rates)
                # find the syllable sung at that rate
                new_syllables.append(
                    neighbor_sylls[neighbor_rates.index(rate)])
                # add error and make sure it's within bounds
                rate += np.random.uniform(-2, 0.25)
                rate = max(rate, 1)
                rate = min(rate, 40)
                new_rates.append(rate)

        for i, xy in enumerate(zip(*self.open_territories)):
            self.bird_matrix[xy[0], xy[1]] = new_syllables[i]
            if self.model_type == 'directional':
                self.rate_matrix[xy[0], xy[1]] = new_rates[i]

    def dispersal(self):
        # which birds are going to move
        swap_matrix = np.array(
            np.random.random((self.dim, self.dim)) < self.dispersal_rate)
        to_disperse = [np.array(i) for i in zip(*np.where(swap_matrix))]
        swap_matrix = swap_matrix.astype(int)

        swap_order = list(np.random.permutation(len(to_disperse)))

        for n, idx in enumerate(swap_order):
            x, y = to_disperse[idx]
            swap_matrix[x, y] = n + 1

        swaps = []
        # swap birds until there are no more to swap
        while swap_order:
            # select the initial bird to disperse
            bird_init = swap_order.pop()
            x, y = to_disperse[bird_init]
            if swap_matrix[x, y] == 0:
                continue

            # mark it as dispersed so it isn't selected again
            swap_matrix[x, y] = 0
            # find adjacent birds
            row_0 = max(0, x - self.dispersal_dist)
            row_1 = min(self.dim, x + self.dispersal_dist + 1)
            col_0 = max(0, y - self.dispersal_dist)
            col_1 = min(self.dim, y + self.dispersal_dist + 1)
            swap_bird = np.max(swap_matrix[row_0:row_1, col_0:col_1])

            if swap_bird != 0:
                swaps.append((bird_init, swap_bird))
                swap_matrix[to_disperse[swap_bird]] = 0

            # check_counter = len(swap_order)
            # while check_counter:
            #     check_counter -= 1
            #     bird_check = swap_order[check_counter]
            #     # check every bird in order; if it's within dispersal distance, swap the birds
            #     if np.all(
            #             to_disperse[bird_check] - bird_init_pos <= self.dispersal_dist):
            #         swap_order.remove(bird_check)
            #         swaps.append((bird_init, bird_check))
            #         check_counter = 0

        for i, j in swaps:
            idx1 = to_disperse[i]
            idx2 = to_disperse[j]
            self.bird_matrix[idx1[0], idx1[1]], self.bird_matrix[idx2[0], idx2[1]] = \
                self.bird_matrix[idx2[0], idx2[1]], self.bird_matrix[idx1[0], idx1[1]]
            if self.model_type == 'directional':
                self.rate_matrix[idx1[0], idx1[1]], self.rate_matrix[idx2[0], idx2[1]] =\
                    self.rate_matrix[idx2[0], idx2[1]], self.rate_matrix[idx1[0], idx1[1]]
            # x1, y1 = to_disperse[idx1]
            # x2, y2 = to_disperse[idx2]
            #
            # syll = self.bird_matrix[x1, y1]
            # self.bird_matrix[x1, y1] = self.bird_matrix[x2, y2]
            # self.bird_matrix[x2, y2] = syll
            #
            # if self.model_type == 'directional':
            #     rate = self.rate_matrix[x1, y1]
            #     self.rate_matrix[x1, y1] = self.rate_matrix[x2, y2]
            #     self.rate_matrix[x2, y2] = rate

    def record_history(self):
        """
        during the sample iterations, record history of the syllable matrix for analysis
        """

        self.history.append(self.bird_matrix.copy())

    def log_stats(self):
        """
        write several values to a log file
        """

        num_syllables, syll_counts = np.unique(self.bird_matrix, return_counts=True)
        num_syllables = len(num_syllables)
        
        mean_syll_count = np.mean(syll_counts)
        std_syll_count = np.std(syll_counts)
        
        with open(self.logfile, 'a') as logfile:
            logfile.writelines("\n{}	{}	{}	{}".format(
                self.generations, num_syllables, mean_syll_count, std_syll_count))

    def time_step(self):
        """
        This has the following steps:
        - Death, where birds are removed at a freq. of 40%
        - Get new syllables from living neighbors based on biases
        - Fill open territories
        - Dispersal of adults
        - Record history, if the iterations need to be sampled (68, after 932 burn-in)
        """
        self.death_step()
        self.fill_open(distance=1)
        if self.dispersal_rate > 0:
            self.dispersal()
        if self.generations >= self.generations_for_burnin:
            self.record_history()

        self.generations += 1
        self.log_stats()

        return 0

    def run(self, generations):
        """
        run the model for x generations and return `history`
        :type generations: int
        :param generations: Number of generations to run the model
        :return history: List containing the syllable matrices from \
        the final iterations, used later for sampling
        """
        for gen in range(generations):
            self.time_step()

        return self.history
