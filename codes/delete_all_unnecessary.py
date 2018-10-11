import os
import shutil
import sys

sys.stdout = open("deletion_result.txt", "a")

for name in range(1500):
    directory_list = next(os.walk('simulation_data/' + str(name)))[1]
    directory_no = len(directory_list)
    if (directory_no != 47) and (directory_no != 44):
        print(name)
        shutil.rmtree('simulation_data/' + str(name))
    else:
        directory_list.remove('VTK')
        for something in directory_list:
            shutil.rmtree('simulation_data/' + str(name) + '/' + something)

sys.stdout.close()