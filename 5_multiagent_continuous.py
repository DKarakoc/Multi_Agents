import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import math

import scipy.optimize as opt

import copy
import os
import sys

map_size = [40,40]
x = np.arange(map_size[0])
y = np.arange(map_size[1])
X, Y = np.meshgrid(x, y)

class AgentContinuos():

    # Constructor
    def __init__(self, state=np.array([0,0,0.2])):
        self.V = 2  # Velocity of the agent
        self.dt = 1  # Interval for euler integration (continuous case)
        self.max_turn_change = 0.2  # Max angle turn (action bounds)

        self.x = state
        self.track = self.x.T  # Stores all positions
        self.height_plot = 0.1

        self.width = map_size[0]
        self.height = map_size[1]

        ## Sensor params
        self.Pdmax = 0.8  # Max range sensor
        self.dmax = 4  # Max distance
        self.sigma = 0.7  # Sensor spread (standard deviation)

    # set next state
    def next(self, vk):
        # singular case u = 0 -> the integral changes
        if vk == 0:
            self.x[0] = self.x[0] + self.dt * self.V * np.cos(self.x[2])
            self.x[1] = self.x[1] + self.dt * self.V * np.sin(self.x[2])
            self.x[2] = self.x[2]
        else:
            desp = self.V / vk
            if np.isinf(desp) or np.isnan(desp):
                print('forwardstates:V/u -> error');
            self.x[0] = self.x[0] + desp * (np.sin(self.x[2] + vk * self.dt) - np.sin(self.x[2]))
            self.x[1] = self.x[1] + desp * (-np.cos(self.x[2] + vk * self.dt) + np.cos(self.x[2]))
            self.x[2] = self.x[2] + vk * self.dt

        self.track = np.vstack((self.track, self.x))

    def pnd(self, x, i, j):
        d = math.sqrt(math.pow((i - x[0]), 2) + math.pow((j - x[1]), 2))
        pd = self.Pdmax * math.exp(-self.sigma * math.pow(d / self.dmax, 2))
        pnd = 1 - pd
        return pnd

    def update_belief(self, belief):
        prior = belief.reshape((map_size[0], map_size[1]))
        updated_belief = np.zeros((self.width, self.height))

        for i in range(len(updated_belief)):
            for j in range(len(updated_belief)):
                updated_belief[i, j] = prior[i, j] * self.pnd(self.x, i, j)

        # normalize belief
        # norm_factor = sum(sum(updated_belief))  # maybe fix?
        # print('normalization factor: ', norm_factor)
        # belief = updated_belief / norm_factor
        self.bk = updated_belief.flatten()
        return updated_belief.flatten()

    def plot(self, ax):
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2);
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0],
                'ko-', linewidth=2);

a = AgentContinuos()

for i in range(8):
    a.next(0.15)

# plot agent trajectory, self.track
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(a.track[:, 0], a.track[:, 1], np.ones(a.track.shape[0]) * 0.1, 'r-', linewidth=2);
ax.plot([a.track[-1, 0], a.track[-1, 0]], [a.track[-1, 1], a.track[-1, 1]], [0.1, 0], 'ko-', linewidth=2);
plt.show()

def multi_utility(uk, agents, N, bk):
    uk = uk.reshape((N, len(agents)))
    copied_agents = copy.deepcopy(agents)
    copied_bk = copy.deepcopy(bk)
    for j in range(N):
        for i in range(len(agents)):
            agent = copied_agents[i]
            action = uk[j,i]
            agent.next(action)
            copied_bk = agent.update_belief(copied_bk)
    return sum(copied_bk)

class Optimizer:

    def __init__(self):
        self.method = 'trust-constr' # Optimization method
        self.jac = "2-point" # Automatic Jacobian finite differenciation
        self.hess =  opt.SR1() # opt.BFGS()
        self.ul = np.pi / 4  # Max turn constraint for our problem (action limits)

    def optimize(self, fun, x0, agents, N, bk):
        # write your optimization call using scipy.optimize.minimize
        n = x0.shape[0]
        # Define the bounds of the variables in our case the limits of the actions variables
        bounds = opt.Bounds(np.ones(n) * (-self.ul), np.ones(n) * self.ul)
        # minimize the cost function. Note that I added the as arguments the extra variables needed for the function.
        res = opt.minimize(fun, x0, args=(agents, N, bk), method=self.method, jac=self.jac, hess=self.hess, bounds=bounds)
        #  options={'verbose': 1})
        return res


def bivariate_gaussian(mu, Sigma):
    beliefs = np.zeros((map_size[0], map_size[1]))
    for i in range(beliefs.shape[0]):
        for j in range(beliefs.shape[1]):
            division = 1 / math.sqrt((2 * math.pi) ** 2 * np.linalg.det(Sigma))
            exp = math.exp(np.dot(-0.5 * (np.array([i,j]) - mu).T, np.dot(np.linalg.inv(Sigma), (np.array([i,j]) - mu))))
            beliefs[i, j] = division * exp

    beliefs = beliefs / np.sum(beliefs)
    return beliefs

mu = np.array([map_size[0]/2., map_size[1]/2.])# center point
Sigma = np.array([[40,0],[0,60]]) # Bimodal covariance with no dependence.
belief = bivariate_gaussian(mu, Sigma).flatten()

a1 = AgentContinuos(state=np.array([10.,5., 0.2]))
a2 = AgentContinuos(state=np.array([15.,30., 0.2]))
a3 = AgentContinuos(state=np.array([30.,0., 0.2]))

agents=[a1,a2,a3]

# Start algorithm
ite = 0  # iteration count
nite = 50  # number of iterations
found = 0  # target found
N = 2
x0 = np.ones(N * len(agents)) * 0.001
print(x0)
print(x0.shape)
o = Optimizer()

while not found and ite < nite:

    # Compute next best state for each agent given the common belief
    predictions = o.optimize(multi_utility, x0, agents, N, belief)
    print(predictions.x)
    predictions = predictions.x.reshape((N, len(agents)))[0]
    for index, move in enumerate(predictions):
        agents[index].next(move)
        print(agents[index].x)
        belief = agents[index].update_belief(belief)

    # plot
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.contourf(Y, X, belief.reshape((map_size[0], map_size[1])), zdir='z', offset=-0.002, cmap=cm.viridis, alpha=0.5)

    for a in agents:
        a.plot(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('P')
    plt.draw()
    plt.pause(0.1)
    ax.cla()

    # iteration count
    ite += 1

plt.pause(0.1)