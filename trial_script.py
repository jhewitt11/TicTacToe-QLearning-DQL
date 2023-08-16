from collections import defaultdict
from itertools import combinations

import random
import sys
import pickle as pkl
import pandas as pd

from game import policy_selection
from game import ttt_trial


rounds = 10000

# Simple hard coded policies
no_model_policies        = [('random', None), ('greedy', None)]


# Different policies that require a "model" of some type to be used
# Q Table
q_table_models = [0]
q_table_policies = [('q_table', x) for x in q_table_models]


# Combine model dep. policies 
model_policies = q_table_policies


# Combine all policies
policies = model_policies + no_model_policies
combos = list(combinations(policies, 2))


param_l = []
for x_pol, o_pol in combos :
    print(f'Face off : {x_pol}\t{o_pol}')
    param_l.append({
        'policy_x'      : x_pol[0],
        'policy_o'      : o_pol[0],
        'model_id_x'    : x_pol[1],
        'model_id_o'    : o_pol[1],
        'rounds'        : 10000,
    })


results = []
for param in param_l :

    trial = ttt_trial(param)
    trial.run()
    data = trial.get_result_dict()
    results.append(data)

pd.DataFrame(results).to_csv('results/results.csv')