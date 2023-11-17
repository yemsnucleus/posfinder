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

## Data requirements 

Data in `.fits` format should be located in the same folder with the following structure: 

- `center_im.fits`: Cube
- `median_unsat.fits`: PSF
- `rotnth.fits`: Parallactic angles
- `init_guess.toml`: Initial values for the planet

The `init_guess.toml` file must contain the following fields: 

```toml
[sep]
name   = "separation"
values = [344.35538038, 344.35538038] 

[theta]
name   = "parallactic_angles"
values = [257.24436263, 257.24436263]

[flux]
name   = "flux"
values = [5281.53524647, 5281.53524647]
```
Note that the `values` parameter is a list containing the initial values for each wavelength in the cube.
