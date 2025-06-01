import numpy as np
SEED = 0
NUM_PROCESSES = 5 # This creates NUM_PROCESSES-1 for some reason
N_EPISODES = 100 # Number of episodes to run each trial
NUM_SECONDS_PER_EPISODE = 20
RECORDING_PATH = 'tests'

BASE_HEADING_VEL_VECS = [np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([1, 0, 0])]
REF_BASE_LIN_VELS = [1., -1., 1., -1., 0.,  0., 0.]
REF_BASE_ANG_VELS = [0.,  0., 0.,  0., 1., -1., 0.]

N_TRIALS = len(REF_BASE_LIN_VELS) # Number of different things we will try

FRICTION_COEFF = (0.5, 1.0)