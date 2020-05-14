import numpy as np

class Kalman(object):

    def __init__(self):
        self.imagesize = (1280, 720)
        self.reset()

    def reset(self):

        self.X = np.array([ [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0]])
        #because we don't know the inital place, so the initial uncertainty is very large
        self.P = np.array([ [500, 0, 0, 0, 0, 0],
                            [0, 500, 0, 0, 0, 0],
                            [0, 0, 500, 0, 0, 0],
                            [0, 0, 0, 500, 0, 0],
                            [0, 0, 0, 0, 500, 0],
                            [0, 0, 0, 0, 0, 500]])

        self.u = np.zeros((6,1))
        self.F = np.array([[1, 1, 0.5, 0, 0, 0], # x, x., x..
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0.5], # y ,y., y..
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]])
        self.R = np.array([[10, 0],
                            [0, 10]])
        self.I = np.eye(6)

    def update(self, Z):
        u_exp = self.H @ self.X
        s2_exp=self.H @ self.P @ self.H.T

        K = self.P @ self.H.T @ np.linalg.pinv(s2_exp+self.R)
        self.X = self.X + K @ (Z-u_exp)
        self.P = (self.I-K @ self.H) @ self.P 
    
    def predict(self):
        self.X = self.F @ self.X+self.u
        self.P = self.F @ self.P @ self.F.T
        return self.X