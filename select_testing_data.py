"""
This file is responsible for randomly selecting users from the 480,189 users to train and test a
model for.

@author: David Rocker
@author: Pranshu Sinha
@author: Sameer Poudwal
"""
from sklearn.model_selection import train_test_split
import pickle as pkl

# Open the qualifying test set
with open('qualifying.txt', 'r') as f:
        lines = f.readlines()
        
# Read in the data
movies = []
data = []
for line in lines:
    if ":" in line:
        movie_id = int(line[:-2])
    else:
        l = line.split(',')
        movies.append(movie_id)
        data.append([int(l[0]), l[1]])

# Randomly select a specific number of users
X_train, X_test, y_train, y_test = train_test_split(data, movies, test_size=0.01, random_state=42)


x = set(y_test)
print(len(x))

users = [x[0] for x in X_test]
users.sort()
x = set(users)
print(len(x))

movies = list(set(movies))
movies.sort()

# Save the user and movie id lists for later
with open('userid_list.pkl', 'wb') as f:
    pkl.dump(users, f)
    
with open('movieid_list.pkl', 'wb') as f:
    pkl.dump(movies, f)

