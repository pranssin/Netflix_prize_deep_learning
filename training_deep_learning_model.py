"""
This file is tasked with:
    * Creating and training our model

@author: David Rocker
@author: Pranshu Sinha
@author: Sameer Poudwal
"""

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Merge, Activation
import numpy as np
import pickle as pkl

# Import the hash table that maps the movie ID to its average rating
with open('movie_avg_dictionary.pkl', 'rb') as f:
    movie_avg_dict = pkl.load(f)

# Import user coefficients from the user-specific regression
with open('user_coef_data_3.pkl', 'rb') as f:
    user_coef = pkl.load(f)
    
with open('user_coef_data_4.pkl', 'rb') as f:
    user_coef2 = pkl.load(f)

# Import the input training data
with open('x_train_new.pkl', 'rb') as f:
    x_train = pkl.load(f)

# Import the output for the training data
with open('y_train_new.pkl', 'rb') as f:
    y_train = pkl.load(f)
    
# Combine the two user coefficient coefficient hash tables
user_coef.update(user_coef2)

# Get the movie average and user rating (from the regression) for each row/entry in the training data
reg_list = []
movie_avg = []
for row in x_train.itertuples():
    reg = row.date * user_coef[row.user_id][2] + user_coef[row.user_id][1]
    reg_list.append(reg)
    movie_avg.append(movie_avg_dict[row.movie_id])
    

# Create our neural network
movie_count = 17771
user_count = 2649430

# Create all 4 embeddings/models
model_left = Sequential()
model_left.add(Embedding(movie_count, 60, input_length=1))

model_right = Sequential()
model_right.add(Embedding(user_count, 20, input_length=1))

model_movierating = Sequential()
model_movierating.add(Embedding(10, 20, input_length=1))

model_userrating = Sequential()
model_userrating.add(Embedding(10, 20, input_length=1))

# Merge the models together
model = Sequential()
model.add(Merge([model_left, model_right, model_movierating, model_userrating], mode='concat'))
model.add(Flatten())

# Add the 3 intermediate layers
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adadelta')

# Put our input data into individual arrays to be used by the model
L = len(x_train['movie_id'].values)
movies = x_train['movie_id'].values.reshape((L,1))
user_ids = x_train['user_id'].values.reshape((L,1))
output = y_train.reshape((L,1))

movie_avg = np.array(movie_avg)
movie_avg = movie_avg.reshape((L, 1))
reg_list = np.array(reg_list)
reg_list = reg_list.reshape((L, 1))


# Import the testing data for our validation dataset in the same was as the input data
with open('x_test_new.pkl', 'rb') as f:
    x_test = pkl.load(f)

with open('y_test_new.pkl', 'rb') as f:
    y_test = pkl.load(f)

# Get the movie average and user rating (from the regression) for each row/entry in the testing data
val_reg_list = []
val_movie_avg = []
for row in x_test.itertuples():
    reg = row.date * user_coef[row.user_id][2] + user_coef[row.user_id][1]
    val_reg_list.append(reg)
    val_movie_avg.append(movie_avg_dict[row.movie_id])


L = len(x_test['movie_id'].values)
val_movies = x_test['movie_id'].values.reshape((L,1))
val_user_ids = x_test['user_id'].values.reshape((L,1))
val_output = y_test.reshape((L,1))

val_movie_avg = np.array(val_movie_avg)
val_movie_avg = val_movie_avg.reshape((L, 1))
val_reg_list = np.array(val_reg_list)
val_reg_list = val_reg_list.reshape((L, 1))

'''
Train the model for the specified number of iterations/epochs.
Save the model after each iteration/epoch.
Validate the model with the validation dataset at each iteration/epoch.
'''
for i in range(0, 30):
    model.fit([movies, user_ids, movie_avg, reg_list], output, batch_size=1500, epochs=1, 
              validation_data=([val_movies, val_user_ids, val_movie_avg, val_reg_list], val_output))
    
    model.save('Netflix_DeepLearning_Model_adadelta.h5')
    
    print("Saved model at loop iteration {}".format(i))


