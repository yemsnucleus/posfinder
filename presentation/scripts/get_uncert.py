#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys
import pandas as pd

DATADIR  = sys.argv[1]
TARGDIR  = './data/records/'

try:
    NRUNS = int(sys.argv[2])
except:
    NRUNS = 5

root = 'python -m presentation.scripts.get_position --data {} --target {} --model {}'

for model in ['gauss']:
    for run in range(NRUNS):
        print('[INFO] Running {}-trial on {} model'.format(run, model))
        command1 = root.format(DATADIR, TARGDIR, model)
        try:
            # subprocess.call(command1, shell=True)
            process = subprocess.run(command1, check=True, shell=True, stdout=subprocess.PIPE)
            output = process.stdout.decode()
            print(output)
        except Exception as e:
            print(e)
