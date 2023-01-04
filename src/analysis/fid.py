import numpy as np
import scipy.linalg

def compute_fid(samples_1, samples_2):
	"""
	Computes the Frechet inception distance between two samples of objects.
	Although the two arrays can have any shape (number of dimensions must be at
	least 2), they must be the same shape and distance will be computed by
	flattening anything after the batch dimension.
	Arguments:
		`samples_1`: a B x ... array of B samples
		`samples_2`: a B' x ... array of B' samples, in the same shape as
			`samples_1`
	Returns a scalar FID score.
	"""
	assert samples_1.shape[1:] == samples_2.shape[1:]
	assert len(samples_1.shape) >= 2
	
	samples_1 = np.reshape(samples_1, (samples_1.shape[0], -1))
	samples_2 = np.reshape(samples_2, (samples_2.shape[0], -1))
	# Shape: B x D
	
	mean_1, cov_1 = np.mean(samples_1, axis=0), np.cov(np.transpose(samples_1))
	mean_2, cov_2 = np.mean(samples_2, axis=0), np.cov(np.transpose(samples_2))
	# Mean shape: D; Covariance matrix shape: D x D
	
	mean_square_norm = np.sum(np.square(mean_1 - mean_2))
	
	# Compute covariance term; because covariance matrices are symmetric,
	# the order of multiplication here doesn't matter
	cov_mean = scipy.linalg.sqrtm(np.matmul(cov_1, cov_2))
	if np.iscomplexobj(cov_mean):
		# Disscard imaginary part if needed
		cov_mean = np.real(cov_mean)
	cov_term = np.trace(cov_1 + cov_2 - (2 * cov_mean))
	
	return mean_square_norm + cov_term
