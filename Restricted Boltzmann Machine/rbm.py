import numpy as np
from scipy.special import expit as sigmoid

class RBM:
	def __init__(self, nv, nh):
		# weights
		self.W = np.random.randn(nv, nh)

		# biases
		self.a = np.random.randn(1, nh)
		self.b = np.random.randn(1, nv)

	def __sample_vh(self, x):
		y = np.dot(x, self.W) + np.repeat(self.a, len(x), axis=0)
		y = sigmoid(y)
		return y, (np.random.rand(*y.shape) < y) * 1

	def __sample_hv(self, y):
		x = np.dot(y, self.W.T) + np.repeat(self.b, len(y), axis=0)
		x = sigmoid(x)
		return x, (np.random.rand(*x.shape) < x) * 1

	def train(self, data, epochs=10, batch_size=1, k_steps=10, output=True):
		batches = np.ceil(len(data) / batch_size)
		data = np.array_split(data, batches)

		for epoch in range(epochs):
			if output:
				print('EPOCH: %d, ' % (epoch + 1), end='');
			
			loss = 0
			for x in data:
				x_valid = (x == 0) | (x == 1)

				x0 = x
				xk = x
				p0,_ = self.__sample_vh(x0)
				
				for k in range(k_steps):
					_,yk = self.__sample_vh(xk)
					_,xk = self.__sample_hv(yk)
					xk[~x_valid] = x0[~x_valid]
				
				pk,_ = self.__sample_vh(xk)
				loss += np.mean(np.abs(x0[x_valid] - xk[x_valid]))

				# adjust weights
				self.W += np.dot(x0.T, p0) - np.dot(xk.T, pk)
				self.b += np.sum(x0 - xk, axis=0)
				self.a += np.sum(p0 - pk, axis=0)
			
			if output:
				print('loss: %f' % (loss / len(data)))

		return loss

	def predict(self, x):
		hv = self.__sample_hv
		vh = self.__sample_vh

		x_valid = (x == 0) | (x == 1)
		res = hv(vh(x)[1])[1]

		loss = np.mean(np.abs(x[x_valid] - res[x_valid]))

		return res,loss
