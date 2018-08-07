import numpy as np
from scipy.special import expit as sigmoid
from tqdm import tqdm

class RBM:
	def __init__(self, nv, nh):
		# weights
		self.W = np.random.randn(nv, nh)

		# biases
		self.a = np.random.randn(1, nh)
		self.b = np.random.randn(1, nv)

	def __sample_uniform(self, u):
		return (np.random.rand(*u.shape) < u) * 1

	def __sample_vh(self, x):
		y = np.dot(x, self.W) + np.repeat(self.a, len(x), axis=0)
		y = sigmoid(y)
		return y

	def __sample_hv(self, y):
		x = np.dot(y, self.W.T) + np.repeat(self.b, len(y), axis=0)
		x = sigmoid(x)
		return x

	def train(self, data, epochs=10, batch_size=1, k_steps=10, output=True):
		batches = np.ceil(len(data) / batch_size)
		data = np.array_split(data, batches)

		sample_uniform = self.__sample_uniform
		vh = self.__sample_vh
		hv = self.__sample_hv

		for epoch in range(epochs):
			loss = 0
			
			if output:
				pbar = tqdm(data, desc=('Epoch %3d' % (epoch+1)))
			else:
				pbar = data

			for x in pbar:
				x_valid = (x == 0) | (x == 1)

				x0,xk = x,x
				p0 = vh(x0)
				
				for k in range(k_steps):
					xk = sample_uniform(hv(sample_uniform(vh(xk))))
					xk[~x_valid] = x0[~x_valid]
				
				pk = vh(xk)
				loss += np.mean(np.abs(x0[x_valid] - xk[x_valid]))

				self.W += np.dot(x0.T, p0) - np.dot(xk.T, pk)
				self.b += np.sum(x0 - xk, axis=0)
				self.a += np.sum(p0 - pk, axis=0)
			
			if output:
				print('Loss %.3f' % (loss / len(data)))

		return loss

	def predict(self, x):
		sample_uniform = self.__sample_uniform
		hv = self.__sample_hv
		vh = self.__sample_vh

		x_valid = (x == 0) | (x == 1)
		res = sample_uniform(hv(sample_uniform(vh(x))))

		loss = np.mean(np.abs(x[x_valid] - res[x_valid]))

		return res,loss
