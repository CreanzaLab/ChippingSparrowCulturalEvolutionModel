import os
import re
import gzip
import lzma
import pickle
import pandas as pd
import numpy as np
import FisherExact
from scipy.stats import chisquare, anderson_ksamp, ks_2samp
from combine_recording_data import load_recording_data

import rpy2.robjects.numpy2ri
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
r_stats = importr('stats')

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('dim', type=int, default=500,
                    help="Size of the model matrix. ")
# parser.add_argument('simulation_number', type=int, default=0,
#                     help="Number of this simulation's run")
parser.add_argument('return_spectra', type=bool, default=False)

args = parser.parse_args()


def syll_spectra(sylls):
    # input: R x 2 matrix with R syllables, with 
    #  their ID in the first column and their
    #  year in the second.
    # output: counts and lifetimes for each syllable
    lifetimes = []
    unique_sylls, syll_counts = np.unique(sylls[:, 0], return_counts=True)

    for syll in unique_sylls:
        years = sylls[sylls[:, 0] == syll, 1]
        lifetimes.append(int(max(years) - min(years) + 1))

    lifetimes = np.array(lifetimes, dtype=int)
    return syll_counts.astype(int), lifetimes.astype(int)


def get_bins(syll_dict):
    # get a dictionary with 
    #  keys = an observed count or lifetime
    #  values = number of times this was found
    return np.arange(1, 2 + max(syll_dict.keys()))


def collapse_spectrum(syllables, counts_or_lifetimes=None, total_num_samples=820):
    # collapses bins of counts or lifetimes to make them more comparable
    # input: either dict or list
    #    dict {(count or lifetime):(# of samples) for each count/lifetime}
    #    list(count or lifetime for every sample)
    # output:
    #    dict {(binned count or lifetime):(total # of samples in that bin) for each count/lifetime bin}
    #    list(count or lifetime bin for every sample)

    if (counts_or_lifetimes == "counts") or (counts_or_lifetimes == "c"):
        bin_edges = [
            1,
            max(2, np.floor(total_num_samples/250)),
            max(3, np.floor(total_num_samples/160)),
            max(4, np.floor(total_num_samples/80)),
            max(5, np.floor(total_num_samples/50)),
            max(6, np.floor(total_num_samples/32)),
            total_num_samples
        ]
    if (counts_or_lifetimes == "lifetimes") or (counts_or_lifetimes == "l"):
        bin_edges = [
            1, 2, 20, 35, 49, 57, 61, 69
        ]

    bins = {
        i:np.arange(bin_edges[i], bin_edges[i+1])
        for i in range(len(bin_edges) - 1)
    }

    collapsed_spectrum = None
    to_delete = []
    if type(syllables) is dict:
        collapsed_spectrum = bins.copy()
        for i in range(len(bins)):
            collapsed_spectrum[i] = 0
            for b in bins[i]:
                if b in syllables:
                    collapsed_spectrum[i] += syllables[b]
            if collapsed_spectrum[i] == 0:
                to_delete.append(i)
    # get rid of 0's in the syllables table
    to_delete.reverse()
    for i in to_delete:
        del collapsed_spectrum[i]

    if type(syllables) is np.ndarray:
        syllables = syllables.tolist()    
    if type(syllables) is list:
        collapsed_spectrum = np.array(syllables)
        for i in range(len(bins)):
            for b in bins[i]:
                collapsed_spectrum[collapsed_spectrum == b] = i
        collapsed_spectrum = collapsed_spectrum.tolist()

    return collapsed_spectrum


# get total data and create tables of syllables from three areas
tab = load_recording_data()
get_sampling_freq = lambda x: x.RecordingYear.value_counts(
    ).sort_index().reindex(range(1950, 2018, 1)).fillna(0).astype(int).values

ny_area = tab[
    (tab.Longitude > -77) *
    (tab.Longitude < -73) *
    (tab.Latitude < 43) *
    (tab.Latitude > 40)]

oh_mi_area = tab[
    (tab.Longitude > -85) *
    (tab.Longitude < -82) *
    (tab.Latitude < 43) *
    (tab.Latitude > 39)]

#ca_centval = tab[
#    (tab.Longitude > -122) *
#    (tab.Longitude < -118) *
#    (tab.Latitude < 39) *
#    (tab.Latitude > 34)]

ne_area = tab[
    (tab.Longitude > -74) *
    (tab.Longitude < -69) *
    (tab.Latitude < 45) *
    (tab.Latitude > 41)
]

areas = (tab, ny_area, oh_mi_area, ne_area)
area_names = ("All", "NY", "OH/MI", "New England")

output_counter = 1
output_df = pd.DataFrame(columns=["model_type", "dim",
                                  "high_syll_rate", "learning_error",
                                  "dispersal_rate", "resampling", "area",
                                  "counts_fisher", "counts_fisher_R",
                                  "counts_fisher_exact_bool", "counts_chi_sq", "counts_chi_sq_p",
                                  "counts_ks", "counts_anderson_k",
                                  "lifetimes_fisher", "lifetimes_fisher_R",
                                  "lifetimes_fisher_exact_bool", "lifetimes_chi_sq",
                                  "lifetimes_chi_sq_p", "lifetimes_ks", "lifetimes_anderson_k"])

dim = args.dim
high_syll_type = 500  # int(dim ** 2 / 500)
resampling = 50


# sim_num = ""
# if args.simulation_number:
#     sim_num = "sim{}".format(args.simulation_number)


count_dict = {}
count_binned_dict = {}
lifetime_dict = {}
lifetime_binned_dict = {}
count_lifetime_lists = {
    "count_dict": count_dict,
    "count_binned_dict": count_binned_dict,
    "lifetime_dict": lifetime_dict,
    "lifetime_binned_dict": lifetime_binned_dict}

for model_type in ("neutral", "conformity", "directional"):
    for error in (0.0001, 0.001, 0.01, 0.1, 1.0):
        for dispersal_rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            
            counts_to_compare = []
            counts_sample_format = []
            lifetimes_to_compare = []
            lifetimes_sample_format = []

            paths = [f"out_dir/{model_type}_{error}err"\
                     f"_1000iters_{dim}dim_{high_syll_type}initSylls"\
                     f"_40mortRate_{dispersal_rate}dispRate"\
                     f".history.pickle"]

            for i in range(1, 100):
                if os.path.exists(re.sub("dispRate", f"dispRatesim{i}", paths[0]) + ".xz"):
                    paths.append(re.sub("dispRate", f"dispRatesim{i}", paths[0]))
#                 else:
#                     print(re.sub("dispRate", f"dispRatesim{i}", paths[0]))
            
            for area, area_name in zip(areas, area_names):
                sampling_freq = get_sampling_freq(area)

                observed_counts_assamples, observed_lifetimes_assamples =\
                    syll_spectra(area.loc[
                    :, ("ClusterNoAdjusted", "RecordingYear")].values)
                count_dict[f"{area_name}"] = observed_counts_assamples
                lifetime_dict[f"{area_name}"] = observed_lifetimes_assamples

                observed_counts_assamples = collapse_spectrum(
                    observed_counts_assamples, "counts", sum(sampling_freq))
                observed_lifetimes_assamples = collapse_spectrum(
                    observed_lifetimes_assamples, "lifetimes")

                count_binned_dict[f"{area_name}"] = observed_counts_assamples
                lifetime_binned_dict[f"{area_name}"] = observed_lifetimes_assamples

                counts_sample_format.append([observed_counts_assamples, []])
                lifetimes_sample_format.append([observed_lifetimes_assamples, []])
                observed_counts = {
                    i:c 
                    for i, c in zip(*np.unique(
                        observed_counts_assamples, return_counts=True))}
                observed_lifetimes = {
                    i:c 
                    for i, c in zip(*np.unique(
                        observed_lifetimes_assamples, return_counts=True))}
                
                counts = {}
                lifetimes = {}
 
                for filepath in paths:
                    model_out = None
                    if os.path.isfile(filepath + ".gz"):
                        with gzip.open(filepath + ".gz", "rb") as f:
                            print(filepath + ".gz")
                            model_out = pickle.load(f)[-68:]
                    if os.path.isfile(filepath + ".xz"):
                        with lzma.open(filepath + ".xz", "rb") as f:
                            print(filepath + ".xz")
                            model_out = pickle.load(f)[-68:]
                    if model_out is None:
                        continue
    
                    model_out = [m.flatten() for m in model_out]
                
                    for _ in range(resampling):
                        sampling = []
                        # for [num_samples] syllables from every year, get 
                        # [num_samples] syllables from the model output
                        for n, num_samples in enumerate(sampling_freq):
                            if num_samples == 0:
                                sampling.append([])
                            else:
                                sampling.append(np.random.choice(
                                    model_out[n], num_samples, replace=False))
    
                        # create a table with a row for each syllable,
                        #  including its syllable number and year of recording
                        syllables = []
                        for year, s in enumerate(sampling):
                            for syllable in s:
                                syllables.append([syllable, year])
                        syllables = np.array(syllables)
    
                        spectra = syll_spectra(syllables)  # counts and lifetimes from the model data
                        for count in spectra[0]:
                            try:
                                counts[count] += 1
                            except KeyError:
                                counts[count] = 1
                        for lifetime in spectra[1]:
                            try:
                                lifetimes[lifetime] += 1
                            except KeyError:
                                lifetimes[lifetime] = 1
    
                    del sampling, spectra
    

                for key in counts:
                    for i in np.repeat(key, counts[key]):
                        counts_sample_format[-1][1].append(int(i))
                counts_sample_format[-1] = collapse_spectrum(
                    counts_sample_format[-1], "counts", sum(sampling_freq))

                for key in lifetimes:
                    for i in np.repeat(key, lifetimes[key]):
                        lifetimes_sample_format[-1][1].append(int(i))
                lifetimes_sample_format[-1] = collapse_spectrum(
                    lifetimes_sample_format[-1], "lifetimes")

                count_dict[f"{model_type}_{error}err_{dispersal_rate}dispRate_{area_name}"] = counts
                counts = collapse_spectrum(counts, "counts", sum(sampling_freq))
                count_binned_dict[f"{model_type}_{error}err_{dispersal_rate}dispRate_{area_name}"] = counts

                lifetime_dict[f"{model_type}_{error}err_{dispersal_rate}dispRate_{area_name}"] = lifetimes
                lifetimes = collapse_spectrum(lifetimes, "lifetimes")
                lifetime_binned_dict[f"{model_type}_{error}err_{dispersal_rate}dispRate_{area_name}"] = lifetimes

                unique_counts = np.union1d(
                    list(counts.keys()),
                    list(observed_counts.keys()))

                counts_to_compare.append(
                    np.c_[[observed_counts[i]
                           if i in observed_counts
                           else 0
                           for i in unique_counts],
                          [counts[i]
                           if i in counts
                           else 0
                           for i in unique_counts]])

                unique_lifetimes = np.union1d(
                    list(lifetimes.keys()),
                    list(observed_lifetimes.keys()))

                lifetimes_to_compare.append(
                    np.c_[[observed_lifetimes[i]
                           if i in observed_lifetimes
                           else 0
                           for i in unique_lifetimes],
                          [lifetimes[i]
                           if i in lifetimes
                           else 0
                           for i in unique_lifetimes]])

            del model_out

            if args.return_spectra:
                with open(f"syll_spectra_{dim}.pickle", 'wb') as f:
                    pickle.dump(count_lifetime_lists, f)
            else:
                # find p-values for the observed vs known counts
                for i, area_name in enumerate(area_names):
                    c_comp = counts_to_compare[i]
                    # setup the row in the output table
                    output_df.loc[output_counter] = [
                        model_type, dim, high_syll_type, error,
                        dispersal_rate, resampling, area_name,
                        -1, -1, True, -1, -1, -1, -1,
                        -1, -1, True, -1, -1, -1, -1]
                    output_df.loc[output_counter, "counts"] = c_comp
                    # calculate chisq and its associated p-value
                    chisq, p_val = chisquare(c_comp[:, 1] / resampling, c_comp[:, 0])
                    output_df.loc[output_counter, "counts_chi_sq"] = chisq
                    output_df.loc[output_counter, "counts_chi_sq_p"] = p_val
                    output_df.loc[output_counter, "counts_ks"] = ks_2samp(
                        counts_sample_format[i][0], counts_sample_format[i][1])[1]
                    output_df.loc[output_counter, "counts_anderson_k"] = anderson_ksamp(counts_sample_format[i])[2]
                    print(c_comp)
    
                    # run the fisher test, first in R and then in Python
                    # c_comp[:, 1] = np.round(c_comp[:, 1] / resampling)
                    try:
                        fisher_p = r_stats.fisher_test(c_comp.T, workspace=200000000)[0][0]
                        output_df.loc[output_counter, "counts_fisher_exact_bool"] = True
                    except RRuntimeError:
                        fisher_p = r_stats.fisher_test(c_comp.T, workspace=200000000, simulate_p_value=True, B=1000000)[0][0]
                        output_df.loc[output_counter, "counts_fisher_exact_bool"] = False
                    output_df.loc[output_counter, "counts_fisher_R"] = fisher_p
                    output_df.loc[output_counter, "counts_fisher"] = FisherExact.fisher_exact(c_comp)
                    output_counter += 1
    
                # find p-values for the observed vs known lifetimes
                output_counter -= len(areas)
                for i, area_name in enumerate(area_names):
                    l_comp = lifetimes_to_compare[i]
                    output_df.loc[output_counter, "lifetimes"] = l_comp
                    # calculate chisq and its associated p-value
                    chisq, p_val = chisquare(l_comp[:, 1] / resampling, l_comp[:, 0])
                    output_df.loc[output_counter, "lifetimes_chi_sq"] = chisq
                    output_df.loc[output_counter, "lifetimes_chi_sq_p"] = p_val
                    output_df.loc[output_counter, "lifetimes_ks"] = ks_2samp(
                        lifetimes_sample_format[i][0], lifetimes_sample_format[i][1])[1]
                    output_df.loc[output_counter, "lifetimes_anderson_k"] = anderson_ksamp(lifetimes_sample_format[i])[2]
    
                    # l_comp[:, 1] = np.round(l_comp[:, 1] / resampling)
    
                    # run the fisher test, first in R and then in Python
                    print(l_comp)
                    try:
                        fisher_p = r_stats.fisher_test(l_comp.T, workspace=200000000)[0][0]
                        output_df.loc[output_counter, "lifetimes_fisher_exact_bool"] = True
                    except RRuntimeError:
                        fisher_p = r_stats.fisher_test(l_comp.T, workspace=200000000, simulate_p_value=True, B=1000000)[0][0]
                        output_df.loc[output_counter, "lifetimes_fisher_exact_bool"] = False
                    output_df.loc[output_counter, "lifetimes_fisher_R"] = fisher_p
                    output_df.loc[output_counter, "lifetimes_fisher"] = FisherExact.fisher_exact(l_comp)
                    output_counter += 1
                print(output_df)
    
                output_df.to_csv(f"fisher_{dim}dim.csv")
