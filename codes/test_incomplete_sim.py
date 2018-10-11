"""
Despite of deleting all incomplete simulation by delete_all_unnecessary.py there are still some incomplete simulations contains in VTK folder.
This scripts find all those directory and then delete them manually.
"""

import os
from tqdm import tqdm
directory_list = next(os.walk('simulation_data'))[1]
for sim_no in tqdm(directory_list):
    DIR = './simulation_data/' + sim_no + '/VTK'
    if len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) != 41:
        print(sim_no)
