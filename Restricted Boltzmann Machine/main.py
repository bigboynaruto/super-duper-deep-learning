from rbm import RBM
import numpy as np

# a very dumb dataset
# either 100..0 or 011..1
def get_simple_dataset(size=1000):
	training_set = np.zeros(shape=(size, 10))
	for row in training_set:
		if np.random.randn() < 0.5:
			row[1:] = 1
		else:
			row[0] = 1

	test_set = np.zeros(shape=(2,10))
	test_set[0][0] = 1
	test_set[1][1:] = 1

	return training_set, test_set

training_set,test_set = get_simple_dataset()

r = RBM(training_set.shape[1], 16)
r.train(training_set, epochs=10, batch_size=10, k_steps=10)

print('Test loss: %s' % (r.predict(test_set)[1]))
