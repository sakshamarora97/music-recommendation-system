import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


#Loading the user song rating dataframe
user_song_rating_csv_path = "/Users/admin/Documents/GeorgiaTech-MSA/Academics/Courses/Fall22/CSE6242/Project/Dataset/lastfm-dataset-1K/user_song_interacting.csv"
df_user_song_rating = pd.read_csv(user_song_rating_csv_path).reset_index(drop = True)
df_user_song_rating.drop(columns = ["Unnamed: 0"], inplace=True)

# index users and songs by integers
user2int = {u:i for i, u in enumerate(np.unique(df_user_song_rating['user_idx']))}
songs2int = {m:i for i, m in enumerate(np.unique(df_user_song_rating['song_idx']))}

# get saved user list and corresponding movie lists
saved_users = df_user_song_rating['user_idx'].to_list()
saved_songs = df_user_song_rating['song_idx'].to_list()

# get row and column indices where we need populate 1's
usersidx = [user2int[u] for u in saved_users]
songsidx = [songs2int[m] for m in saved_songs]

# Here, we only use binary flag for data. 1 for all saved instances.
# Instead, one could also use something like count of saves etc.
# data = np.ones(len(saved_users), ) 
data = np.array(df_user_song_rating['Rating']).reshape((len(usersidx),))

# create csr matrix
A_sparse = csr_matrix((data, (usersidx, songsidx)))

print("Sparse array", A_sparse)

#<3x3 sparse matrix of type '<class 'numpy.float64'>'
#   with 4 stored elements in Compressed Sparse Row format>

print(A_sparse.data.nbytes)
# 32

print("Dense array", A_sparse.A)

#array([[1., 1., 0.],
#       [0., 1., 0.],
#       [0., 0., 1.]])

# print(A_sparse.A.nbytes)
# 72

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T

R = A_sparse
R_arr = A_sparse.toarray()
# R = np.array(R)
# N: num of User
U = R_arr.shape[0]
# M: num of songs
S = R_arr.shape[1]

# Num of Features
K = 3

 
P = np.random.rand(U,K)
Q = np.random.rand(S,K)

nP, nQ = matrix_factorization(R_arr, P, Q, K)

# import numpy as np

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

R = R_arr
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)

mf.train()
mf.P
mf.Q

