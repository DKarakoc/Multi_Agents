import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import math

import scipy.optimize as opt

import os
import sys


class Agent:

    # Constructor
    def __init__(self, i, state=np.array([0., 0.]), map_size=np.array([40, 40]), bk=np.ones(1600) * (1. / 1600), offset=-0.002):

        self.id = i  # Create an instance variable
        self.map_size = map_size

        ## Environment. Tip it would be efficient to compute the grid into one vector as we did in the previous notebook

        self.bk = np.reshape(bk, (40, 40))

        ## Agent params
        self.x = state
        self.track = self.x.T  # Stores all positions. Aux variable for visualization of the agent path
        self.height_plot = 0.1

        # add extra agents params
        self.width = map_size[0]
        self.height = map_size[1]
        self.moves = np.array(
            [[-1., -1.], [-1., 0.], [-1., 1.], [0., -1.], [0., 0.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]])
        x = np.arange(map_size[0])
        y = np.arange(map_size[1])
        self.X, self.Y = np.meshgrid(x, y)
        self.offset = offset

        ## Sensor params
        self.Pdmax = 0.8  # Max range sensor
        self.dmax = 4  # Max distance
        self.sigma = 0.7  # Sensor spread (standard deviation)

    # get id
    def get_id(self):
        return self.id

    # compute discrete forward states 1 step ahead
    def forward(self):
        forward = self.moves + self.x
        forward = np.random.permutation(forward)
        fs = []
        for element in forward:
            if element[0] >= 0 and element[0] < (self.width):
                if element[1] >= 0 and element[1] < self.height:
                    fs.append(element)
        return fs

    # computes utility of forward states
    def nodetect_observation(self, x, i, j):
        d = math.sqrt(math.pow((i - x[0]), 2) + math.pow((j - x[1]), 2))
        pd = self.Pdmax * math.exp(-self.sigma * math.pow(d / self.dmax, 2))
        pnd = 1 - pd
        return pnd

    # computes utility of states
    def utility(self, state):
        # compute cost funtion J of all potential forward states (brute force method)
        map = np.zeros((self.map_size[0], self.map_size[1]))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                pnd = self.nodetect_observation(state, i, j)
                b = self.bk[i, j]
                map[i, j] = pnd * b
        return sum(sum(map))

    # simulate agent next state
    def next(self, state):
        self.x = state
        inv = [self.x[1], self.x[0]]
        self.track = np.vstack((self.track, inv))

    # update belief with observation at state self.x
    def update_belief(self):
        prior = self.bk
        updated_belief = np.zeros((self.width, self.height))

        for i in range(len(updated_belief)):
            for j in range(len(updated_belief)):
                updated_belief[i, j] = prior[i, j] * self.nodetect_observation(self.x, i, j)

        # normalize belief
        norm_factor = sum(sum(updated_belief))  # maybe fix?
        # print('normalization factor: ', norm_factor)
        normalized_map = updated_belief / norm_factor
        self.bk = normalized_map

    def next_best_state(self):
        fs = self.forward()
        utilities = []
        for state in fs:
            util = self.utility(state)
            utilities.append(util)
        min_index = np.argmin(utilities)
        return fs[min_index]

    def plot(self, ax):
        # Reshape belief for plotting
        #         bkx = self.bk.reshape((self.map_size[1], self.map_size[0]))
        ax.cla()  # clear axis plot

        # plot contour of the P(\tau) -> self.bk
        ax.contourf(self.X, self.Y, self.bk, zdir='z', offset=-0.002, alpha=0.5, cmap=cm.viridis)

        # plot agent trajectory, self.track
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2);
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0],
                'ko-', linewidth=2);

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('P')

        # Adjust the limits, ticks and view angle
        ax.set_zlim(0, 0.12)
        ax.view_init(27, -21)

        plt.draw()
        plt.pause(0.1)  # animation

    def plot2(self, ax):
        ax.contourf(self.X, self.Y, self.bk, zdir='z', offset=self.offset, alpha=0.8, cmap=cm.viridis)
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2);
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0],
                'ko-', linewidth=2);

    def get_belief(self):
        return self.bk

    def get_track(self):
        return self.track

    def get_state(self):
        return self.x


class Environment:
    def __init__(self):

        self.width = 40
        self.height = 40

        self.x = np.arange(self.width)
        self.y = np.arange(self.width)

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.mu = np.array([self.width / 2., self.width / 2.])
        self.Sigma = np.array([[40, 0], [0, 60]])


        def bivariate_gaussian():
            beliefs = np.zeros((40, 40))
            for i in range(self.width):
                for j in range(self.height):
                    division = 1 / math.sqrt((2 * math.pi) ** 2 * np.linalg.det(self.Sigma))
                    exp = math.exp(np.dot(-0.5 * (np.array([i,j]) - self.mu).T, np.dot(np.linalg.inv(self.Sigma), (np.array([i,j]) - self.mu))))
                    beliefs[i, j] = division * exp

            beliefs = beliefs / np.sum(beliefs)
            return beliefs

        self.beliefs = bivariate_gaussian()

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_size(self):
        return self.X, self.Y

    def get_prior(self):
        return self.beliefs

print('-------------------------------------------------\n');
print('> M-Agents search 2D (1 n-step ahead no comm)\n')

nagents = 2
# Create environment
env = Environment()
# Create agents
agents = []
agents.append(Agent(0, state=np.array([0,0]), bk=env.get_prior().flatten(), offset = -0.002))
agents.append(Agent(1, state=np.array([30,30]), bk=env.get_prior().flatten(), offset = 0.06))

# Global plot for animations
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.ion()

# Start algorithm
ite = 0  # iteration count
nite = 50  # number of iterations
found = 0  # target found


## start search
while not found and ite < nite:
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print('\n', ite)
    for agent in agents:
        print(agent)
        move = agent.next_best_state()
        print('agent state: ', agent.get_state())
        print('move: ', move)
        agent.next(move)
        agent.update_belief()
        X, Y = env.get_size()
        agent.plot2(ax)

    # plot
    plt.draw()
    plt.pause(0.1)  # animation

    # iteration count
    ite += 1
