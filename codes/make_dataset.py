import os
from tqdm import *
import subprocess
import random

num_runs = 1500
x_cord = random.sample(range(100, 2900), num_runs)
y_cord = []

for x in x_cord:
    if x < 500:
        y_limit = 100
        y_cord.append(random.randint(100, y_limit))
    elif 500 <= x < 1500:
        y_limit = 200
        y_cord.append(random.randint(100, y_limit))
    elif 1500 <= x < 2000:
        y_limit = 300
        y_cord.append(random.randint(100, y_limit))
    elif 2000 <= x <= 2900:
        y_limit = 400
        y_cord.append(random.randint(100, y_limit))

x_cord = [round(x / 1000, 2) for x in x_cord]
y_cord = [round(y / 1000, 2) for y in y_cord]


for i in tqdm(range(num_runs)):
    with open(os.devnull, 'w') as devnull:
        # Remove any previous simulation file
        cmd = "rm -rf simulation_data/" + str(i)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Copy the OpenFOAM forwardStep directory
        cmd = "cp -a ./original/. ./simulation_data/" + str(i)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Remove the blockMeshDict file from system directory
        cmd = "rm -f ./simulation_data/" + str(i) + "/system/blockMeshDict"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Execute python program to write a blockMeshDict file
        cmd = "python gen_blockMeshDict.py" + " " + str(x_cord[i]) + " " + str(y_cord[i])
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the cellInformation file to home directory
        cmd = "mv cellInformation ./simulation_data/" + str(i)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
