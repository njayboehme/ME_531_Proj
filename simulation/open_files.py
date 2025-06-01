import pickle

with open('/home/nboehme/Desktop/Quadruped-PyMPC/tests/go2/flat/lin_vel=1.0 ang_vel=0.0 friction=(0.5, 1.0)/ep=10_steps=999/state_hist_0.pkl', 'rb') as f:
    load_data = pickle.load(f)
print(len(load_data))
# print(load_data)