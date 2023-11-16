import matplotlib.pyplot as plt
import numpy as np

def plot_frame(frame, ax=None, n_frames=None):
	
	if n_frames is None:
		n_frames = np.minimum(len(frame), 4)
	if ax is None:
		fig, axes = plt.subplots(1, n_frames)

	try:
		axes = axes.flatten()
	except:
		axes = [axes]

	for index, f in enumerate(frame):
		axes[index].imshow(frame[index], origin='lower')
		axes[index].set_title('{}-frame'.format(index))
	plt.show()
	return axes