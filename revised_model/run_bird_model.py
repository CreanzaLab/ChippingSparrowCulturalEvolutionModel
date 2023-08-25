import os
import sys
import argparse
import pickle
import lzma
import psutil
from birds import BirdModel


parser = argparse.ArgumentParser()

parser.add_argument('model_type', type=str,
                    help='Type of model. One of: "neutral", "conformity", "directional"')
parser.add_argument('dispersal_rate', type=float,
                    help='The amount of dispersal that takes place, as a proportion of the total population.')
parser.add_argument('error_rate', type=float,
                    help='Percentage of birds that learn an incorrect song (and produce a new syllable)')
parser.add_argument('-d', '--dim', type=int, default=500,
                    help='Dimension of the bird matrix.')
parser.add_argument('-s', '--simulation_number', type=int, default=1,
                    help="Number of this simulation's run, when you want to compare multiple runs.")
parser.add_argument('-t', '--total_generations', type=int, default=1000,
                    help='Total number of iterations; burn-in is (TOTAL_ITERATIONS - 68).')
parser.add_argument('--high_syll', type=str, default="constant",
                    help="How the initial number of syllables are calculated. Default is 'constant' "\
                         "which is set to 500. Otherwise, it is 'adaptive' which is set to (dim^2)/500.")
parser.add_argument('--homoplasy', 'homoplasy', action='store_true')
parser.set_defaults(homoplasy=False)

args = parser.parse_args()

"""
Cultural Transmission Model
"""
out_dir = os.getcwd() + '/out_dir'
conformity_factor = 2

iterations = args.total_generations
dim = args.dim

mortality_rate = 0.4
dispersal_dist = 11
low_syll_type = int(0)  # should not change
if args.high_syll == "constant":
    high_syll_type = 500
else:
    int(dim ** 2 / 500)
low_syll_rate = float(5)  # units of syllables/second
high_syll_rate = float(40)  # units of syllables/second

# setup runs with various parameters
file_name = args.model_type + '_' \
            + str(args.error_rate) + 'err_' \
            + str(iterations) + 'iters_' \
            + str(dim) + 'dim_' \
            + str(high_syll_type) + 'initSylls_' \
            + str(int(mortality_rate*100)) + 'mortRate_' \
            + str(args.dispersal_rate) + 'dispRate'
if args.simulation_number > 0:
    file_name = file_name + 'sim' + str(args.simulation_number)
if args.homoplasy:
    file_name = file_name + '_hp'

currently_running = -1
for proc in psutil.process_iter():
    cmd = proc.cmdline()
    if len(cmd) != (len(sys.argv) + 1):
        continue
    if all([i == j for i, j in zip(cmd[1:], sys.argv)]):
        currently_running += 1

if not (os.path.exists(f"{out_dir}/{file_name}.history.pickle.xz") or currently_running):
    with open(f"{out_dir}/{file_name}.log", 'w') as logfile:
        logfile.writelines("iter	num_syll	mean_syll_count	std_syll_count")

    m = BirdModel(args.model_type,
                  args.dispersal_rate,
                  args.error_rate / 100,
                  dim=args.dim,
                  homoplasy=args.homoplasy,
                  logfile=f"{out_dir}/{file_name}.log",
                  total_generations=args.total_generations)
    
    # run for 1000 generations, of which the last 68 are kept for analysis
    counter = 0
    while counter < iterations:
        print(f"{file_name} : iteration:{counter}")
        counter += 100
        history = m.run(100)

    print(f"{file_name} : done")
 
    with lzma.open(f"{out_dir}/{file_name}.history.pickle.xz", 'wb') as f:
        pickle.dump(history, f)
