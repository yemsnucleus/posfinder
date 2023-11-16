import matplotlib.pyplot as plt
import numpy as np

def plot_frame(frame, ax=None, n_frames=None, pos=None):
	
	if n_frames is None:
		n_frames = np.minimum(len(frame), 4)
	if ax is None:
		fig, axes = plt.subplots(1, n_frames, dpi=300)

	try:
		axes = axes.flatten()
	except:
		axes = [axes]

	for index, f in enumerate(frame):
		axes[index].imshow(frame[index], origin='lower')
		axes[index].set_title('{}-frame'.format(index))

		if pos is not None:
			axes[index].scatter(pos[index][0], pos[index][1], marker='x', s=1, color='red')
	plt.show()
	return axes