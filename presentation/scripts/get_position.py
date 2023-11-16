import argparse
import os

from src.hci import load_fits, collapse_to_median
from src.models import gaussian_model
from src.plot import plot_frame

def run(opt):
	data = load_fits(opt.data)

	med_frame = collapse_to_median(data['cube'], 
								   q=data['parallactic'],
								   save=opt.data,
								   overwrite=False)

	estimated_pos = gaussian_model(med_frame, 
								   init_pos=os.path.join(opt.data, 'init_guess.toml'))

	plot_frame(med_frame, pos=estimated_pos)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target', default='./data/records/irdis', type=str,
					help='target folder')
	parser.add_argument('--data', default='./data/clean/', type=str,
					help='Data folder where datasets are found')

	parser.add_argument('--model', default='gauss', type=str,
				help='Model to use')
	

	opt = parser.parse_args()        
	run(opt)
