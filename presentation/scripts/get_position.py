import pandas as pd
import argparse
import time
import os

from src.hci import load_fits, collapse_to_median
from src.models import gaussian_model, gauss_tf_model
from src.plot import plot_frame
from datetime import datetime


def run(opt):

	data = load_fits(opt.data)
	start_time = time.time()
	if opt.reduce == 'median':
		med_frame = collapse_to_median(data['cube'], 
									   q=data['parallactic'],
									   save=opt.data,
									   overwrite=False)
	if opt.model == 'gauss_tf':
		print('[INFO] Gaussian Tensorflow Model')
		estimated_pos = gauss_tf_model(med_frame, 
									   init_pos=os.path.join(opt.data, 'init_guess.toml'))
	if opt.model == 'gauss':	
		estimated_pos = gaussian_model(med_frame, 
									   init_pos=os.path.join(opt.data, 'init_guess.toml'))

	end_time = time.time()
	elapsed_time = end_time - start_time

	# plot_frame(med_frame, pos=estimated_pos)

	os.makedirs(opt.target, exist_ok=True)

	now = datetime.now()
	formatted_time = now.strftime("%m/%d/%y %H:%M:%S")

	rows = []
	for i, fpos in enumerate(estimated_pos):
		rows.append({'filter':data['filters'][i],
					  'model': opt.model,
					  'reduce': opt.reduce,
					  'x': fpos[0], 
					  'y': fpos[1],
					  'elapsed':elapsed_time,
					  'date': formatted_time})

	file_position = os.path.join(opt.target, 'pos.csv')
	df_pos = pd.DataFrame(rows)
	if os.path.isfile(file_position):
		restored = pd.read_csv(file_position)
		df_pos = pd.concat([restored, df_pos], ignore_index=True)
	
	df_pos.to_csv(file_position, index=False)
	print(df_pos)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target', default='./results/eta_tel_b/', type=str,
					help='target folder')
	parser.add_argument('--data', default='./data/clean/', type=str,
					help='Data folder where datasets are found')

	parser.add_argument('--reduce', default='median', type=str,
				help='Reducing technique to be applied')
	parser.add_argument('--model', default='gauss', type=str,
				help='Model to use (gauss, gauss_tf)')
	

	opt = parser.parse_args()        
	run(opt)
