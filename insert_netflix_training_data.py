"""
This file is responsible for inserting all 100+ million records into our database to user later.

@author: David Rocker
@author: Pranshu Sinha
@author: Sameer Poudwal
"""

import os
import database
import pickle as pkl

# Create a general path to where the movie files with the corresponding watch date, user, and rating
# information is located.

#path = "C:\\Users\\David\\Desktop\\CSC522\\Project\\training_set\\"
path = '/home/david/Desktop/CSC 522/Project/Netflix/training_set/'
file_list = os.listdir(path)
file_list.sort()

# Create our database object to connect to our database
datab = database.Database()

# Create a table to store all of our data
create_schema = "CREATE SCHEMA IF NOT EXISTS netflix AUTHORIZATION postgres;"
delete_table = "DROP TABLE IF EXISTS netflix.data;"
create_table = "CREATE TABLE IF NOT EXISTS netflix.data (" \
                      "movie_id integer NOT NULL," \
                      "user_id integer NOT NULL," \
                      "rating integer," \
                      "date_watched date NOT NULL " \
                      ")" \
                      "WITH (" \
                      "     OIDS = FALSE" \
                      ");" \
                      "ALTER TABLE netflix.data OWNER to postgres;"

datab.execute_command(create_schema)
datab.execute_command(delete_table)
datab.execute_command(create_table)


movie_count = []        # List that will store the movie id and the number of users who watched it
count = 0               # Counter for number of users who have watched a movie
insert_count = 0        # Current number of entries we need to insert into the database
insert_max = 1000       # Number of entries needed before we send an insert statement into the database

'''
This loop works by opening a movie file, reading its contents, putting it into an insert statement,
and then inserting the data into the database whenever we hit the insert_max number. The reason for
the insert_max number is that inserting for every rating entry is really slow, whereas doing it for
every insert_max is a lot faster (saves time writing to the database). However, a limit is needed
since trying to do every entry for entire movies at once can cause us to run out of memory.
'''
for file in file_list:
    # open the data for a movie
    file1 = path + file
    with open(file1, 'r') as f:
        l = f.readlines()

    # Get the number of people who have watched the movie
    movie_id = l[0][:-2]
    movie_count.append([movie_id, len(l)])

    # Create a list to store all the insert statements together
    command = []
    for i in range(1, len(l)):
        temp = l[i].split(',')
        command.append("INSERT INTO netflix.data VALUES ({movie},{user},{rating},'{date}');".format(
                movie=movie_id, user=temp[0], rating=temp[1], date=temp[2][:-1]
            ))
        
        # If we have reached the number of insert statements necessary to insert the data into the
        # database, then go ahead and join all the insert commands together and commit it.
        insert_count = insert_count + 1
        if insert_count == insert_max:
            command = "\n".join(command)
            
            try:
                datab.execute_command(command)
                insert_count = 0
                command = []
            except:
                print("Error inserting for file: {}".format(file))

    # If there is any left over data go ahead and commit that into the database as well.
    # There can be extra data if we do not have a multiple of 1000 as the number of entries for a movie
    if insert_count != 0:
        command = "\n".join(command)
        datab.execute_command(command)
        insert_count = 0
        
    print('Finished with movie: {}'.format(movie_id))

# Save the movie count list
with open('movie_count_list.pkl', 'wb') as f:
    pkl.dump(movie_count, f)

