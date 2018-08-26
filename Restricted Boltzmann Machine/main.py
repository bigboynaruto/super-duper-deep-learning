from rbm import RBM
import numpy as np
import pandas as pd

ratings = pd.read_csv('rating.csv', sep=',').values
# throws memory error
ratings = ratings[ratings[:,0] < 20000]

animes = pd.read_csv('anime.csv', sep=',', low_memory=False).values

user_ids = np.unique(ratings[:,0])
anime_ids = np.unique(animes[:,0])

user_count = len(user_ids)
anime_count = len(anime_ids)

user_id_scaler = {user_id:idx for user_id,idx in zip(user_ids, range(user_count))}
anime_id_scaler = {anime_id:idx for anime_id,idx in zip(anime_ids, range(anime_count))}

print('Users: %d, Animes: %d' % (user_count, anime_count))

dataset = np.full((user_count, anime_count), -1)
for row in ratings:
	# seems like ratings.csv contains invalid anime_id
	try:
		dataset[user_id_scaler[row[0]], anime_id_scaler[row[1]]] = row[2]
	except:
		pass

np.random.shuffle(dataset)

dataset[(dataset < 5) & (dataset >= 0)] = 0
dataset[dataset >= 5] = 1

training_set_size = int(dataset.shape[0] * 0.8)
training_set,test_set = dataset[:training_set_size], dataset[training_set_size:]

r = RBM(training_set.shape[1], 16)
r.train(training_set, epochs=10, batch_size=64, k_steps=1)

print('Test loss: %s' % (r.predict(test_set)[1]))

my_input = np.full((1, anime_count), -1)
for row in pd.read_csv('liked.csv', sep=',', low_memory=False).values:
	my_input[:, anime_id_scaler[row[0]]] = 1
for row in pd.read_csv('not_liked.csv', sep=',', low_memory=False).values:
	my_input[:, anime_id_scaler[row[0]]] = 0
	
predicted = r.predict(my_input)[0]
predicted[my_input == 1] = -1
predicted[my_input == 0] = -1
#anime_id = list(anime_id_scaler.keys())[np.argmax(predicted)]
#print('Anime: %s' % (str(animes[animes[:,0]==anime_id])))

sorted = np.argsort(predicted[0])
for idx in sorted[-5:]:
    anime_id = list(anime_id_scaler.keys())[idx]
    print('Anime: %s' % (str(animes[animes[:,0]==anime_id])))
