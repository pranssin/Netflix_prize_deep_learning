"""
This code is responsible for establishing a connection to our PostgreSQL database.

@author: David Rocker
@author: Pranshu Sinha
@author: Sameer Poudwal
"""

from psycopg2 import pool
import pandas as pd


####################################################################################################
#
#                                   Define the connection pool
#
####################################################################################################
# Parameters are as follows:
# * The number of connections to create at the time the connection pool is created
# * Depends on how many simultaneous connections to the database your program needs.
#   If there is a large amount of traffic to/from the database, you need more.
#   It will create more connections as they are needed.
simple_connection_pool = pool.SimpleConnectionPool(1, 1, database="postgres", user="postgres", password="postgres", host="localhost")


####################################################################################################
#
#                       Define a class to work with the connection pool
#
####################################################################################################
class ConnectionPoolHandler:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def __enter__(self):
        self.connection = simple_connection_pool.getconn()
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_value is not None:
            # If this is executed, then an exception did happen
            # e.g. TypeError, AttributeError, etc...
            print("Error commiting to postgre database.")
            print("Exception: ", exception_value)
            self.connection.rollback()
        else:
            self.cursor.close()
            self.connection.commit()
        simple_connection_pool.putconn(self.connection)


####################################################################################################
#
#                       Define a class to handle all database interactions
#
####################################################################################################
class Database:

    def __init__(self):
        pass

    # Command to execute, such as a command to insert data, where no return is expected
    def execute_command(self, command):
        with ConnectionPoolHandler() as cursor:
            try:
                cursor.execute(command)
            except:
                print("Problem executing command.")
                print(command)

    # Command to execute if data is expected to be returned from the SQL query command
    def get_data(self, command, col_list):
        with ConnectionPoolHandler() as cursor:
            cursor.execute(command)
            df = pd.DataFrame(cursor.fetchall(), columns=col_list)
        return df
        
