#!/usr/bin/env python3
import sys
import numpy as np

x_obs = np.arange(100)

y_obs = 0.2* x_obs + 5.0

data = {}
data['a'] = float(sys.argv[1])
data['b'] = float(sys.argv[2])

y_sim = data['a'] * x_obs + data['b']
data['f'] = y_sim

dataName = 'linear_'+ sys.argv[3] + '.npy'
np.save(dataName,data)
