# Routine for getting planet positions

## Installation
Create a virtual environment using conda (or miniconda):
```
conda create -n posfinder python==3.10 notebook
```
Then, inside the virtual environment, install pip dependencies:
```
pip install -r requirements.txt 
```

## How to use
For a single run, use: 
```
python -m presentation.scripts.get_position --data ./my_data_folder --model gauss
```
Use the flag `--help` to see more details about the command-line parameters.

To run several runs and calculate uncertainties, use: 
```
python -m presentation.scripts.get_uncert ./my_data_folder
```
Please note that if needed, you can modify combinations of parameters manually within the `get_uncert` script.