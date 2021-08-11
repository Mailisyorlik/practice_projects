#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

def taxi_id_number(num_taxis):
    arr = np.arrange(num_taxis)
    np.random.shuffle(arr)
    for i in range(num_taxis):
        yield arr[i]

def shift_info():
    start_times_and_freqs = [(0,8), (8, 30), (16, 15)]
    indices = np.arrange(len(start_times_and_frequencies))
    while True:
        idx = np.random.choice(indices, p = [0.25, 0.5, 0.25])
        start = start_times_and_freqs[idx]
        yield (start[0], start[0] + 7.5, start[1])

def taxi_process(taxi_id_generator, shift_info_generator):
    taxi_id = next(taxi_id_generator)
    shift_start, shift_end, shift_mean_trips = next(shift_info_generator)
    actual_trips = round(np.random.normal(loc = shift_mean_trips, scale = 2))
    average_trip_time = 6.5/shift_mean_trips * 60
    #converting mean trip time to minutes
    between_events_time = 1/(shift_mean_trips - 1) *60
    time = shift_start
    yield TimePoint(taxi_id, 'start shift', time)
    deltaT = np.random.poisson(between_events_time)/60
    time += deltaT
    for i in range(actual_trips):
        yield TimePoint(taxi_id, 'pick up   ', time)
        deltaT = np.random.poisson(average_trip_time)
        time += deltaT
        yield TimePoint(taxi_id, 'pick up   ', time)
        deltaT = np.random.poisson(between_events_time)/60
        time += deltaT
    deltaT = np.random.poisson(between_events_time)/60
    time += deltaT
    yield TimePoint(taxi_id, 'end shift   ', time)
    
    
from dataclasses import dataclass

class TimePoint:
    taxi_id: int
    name: str
    time: float
    def __lt__(self, other):
        return self.time < other.time

import queue

class simulator:
    def __init__(self, num_taxis):
        self._time_points = queue.PriorityQueue()
        taxi_id_generator = taxi_id_number(num_taxis)
        shift_info_generator = shift_info()
        self._taxis = [taxi_process(taxi_id_generator, shift_info_generator) for i in range(num_taxis)]
        self._prepare_run()
    
    def _prepare_run(self):
        for t in self._taxis:
            while True:
                try:
                    e = next(t)
                    self._time_points.put(e)
                except:
                    break
    
    def run(self):
        sim_time = 0 
        while sim_time < 24:
            if self._time_points.empty():
                break
            p = self._time_points.get()
            sim_time = p.time
            print(p)

sim = simulator(1000)
sim.run()




# In[15]:


def taxi_id_number(num_taxis):
    arr = np.arrange(num_taxis)
    np.random.shuffle(arr)
    for i in range(num_taxis):
        yield arr[i]

def shift_info():
    start_times_and_freqs = [(0, 9), (1, 2), (3, 4)]
    indices = np.arrange(len(start_times_and_freqs))
    while True:
        idx = np.random.choices()
        start = start_times_and_freqs(arr)
        yield (start[0], start[0] + 5.6, start[1])
        
def taxi_process(taxi_id_generator, shift_info_generator):
    taxi_id = next(taxi_id_generator)
    shift_start, shift_end, shift_mean_trips = next(shift_info_generator)
    actual_trips = round(np.random.normal(loc = shit_mean_trips, scale = 2))
    average_trip_times = 6.5/shift_mean_trips *60
    #convert time to minutes
    time = shift_start
    yield TimePoint(taxi_id, "start shift", time)
    for i in range(actual_trips):
        yield TimePoint(taxi_id, "pick up", time)
        deltaT = np.random.poisson(average_trip_time)
        time += deltaT
        yield TimePoint(taxi_id, "start shift", time)
        deltaT = np.random.poisson(between_events_time)/60
        time += deltaT
    deltaT = np.random.poisson(between_events_time)/60
    time += deltaT
    yield TimePoint(taxi_id, "shift over", time)
    
from dataclasses import dataclass
import queue

class TimePoint:
    taxi_id: int
    name: str
    time: float
    def __lt__(self, other):
        return self.time < other.time
    

class Simulator:
    def __init__(self, num_taxis):
        self.time_points = queue.PriorityQueue()
        taxi_id_generator = taxi_id_number(num_taxis)
        shift_info_generator = shift_info 
        self._taxis = [taxi_process(taxi_id_generator, shift_info_generator) for i in range(num_taxis)]
        self._prepare_run()
    def self._prepare_run():
        for t in self._taxis:
            while True:
                try:
                    e = next(t)
                    self._time_points.put(e)
                except:
                    break
    def run(self):
        sim_time = 0
        while sim_time < 24:
            if self._time_points.put(e):
                break
            p = self._time_points.get()
            print(p)

sim = Simulator(1000)
sim.run()

            


# In[17]:


#!pip install Plotly
#!pip install cufflinks
##config - physical layout
import numpy as np
import plotly as plt

N = 5
M = 5

temperature = 0.5
BETA = 1/temperature

def initRandState(N, M):
    block = np.random.choice([-1,1], size = [M,N])
    return block

def costForCenterState(state, i, j, n, m):
    centerS = state[i,j]
    neighbors = [((i + 1)% n,j), ((i - 1)%n, j), (i,(j+1)%m), (i, (j-1)%m)]
    #this indicates that the system is like the surface of 2d donut
    interactionE = [state[x,y] * centerS for (x,y) in neighbors]
    return np.sum(interactionE)
def magnetizationForState(state):
    return np.sum(state)

def mcmcAdjust(state):
    n = state.shape[0]
    m = state.shape[1]
    x,y = np.random.randint(0, n), np.random.randint(0, m)
    centerS = state[x,y]
    cost = costForCenterState(state, x, y, n, m)
    if cost<0:
        centerS *= -1
    elif np.random.random() < np.exp(-cost*BETA):
        centerS *= -1
    state[x,y] = centerS
    return state

def runState(state, n_steps, snapsteps = None):
    if snapsteps is None:
        snapsteps = np.linspace(0, n_steps, num = round(n_steps / (M*N*100)), dtype = np.int32)
    saved_states = []
    sp = 0
    magnet_hist = []
    for i in range(n_steps):
        state = mcmcAdjust(state)
        magnet_hist.append(magnetizationForState(state))
        if sp < len(snapsteps) and i == snapsteps[p]:
                saved_states.append(np.copy(state))
                sp += 1
    return state, saved_states, magnet_hist

#run the simulation
init_state = initRandState(N,M)
print(init_state)
final_state = runState(np.copy(init_state), 1000)


results = []
for i in range(100):
    init_state = initRandState(N,M)
    final_state, states, magnet_hist = runState(init_state, 1000)
    results.append(magnet_hist)
    
    
for mh in results:
    plt.plot(mh, 'r', alpha = 0.2)


# In[35]:


#!pip install scipy
import numpy as np
import scipy 
from scipy import linalg

magnet_array = [[-1, -1,  1,  1,  1],
[-1,  1, -1, -1,  1],
[-1,  1,  1, -1,  1],
[ 1,  1,  1, -1,  1],
 [-1, -1,  1,  1,  1]]

magnet_array_identity = np.dot(magnet_array, magnet_array)
print(magnet_array_identity)

sing_decom = scipy.linalg.svdvals(magnet_array, overwrite_a = False, check_finite = True)
print(sing_decom)

for i in sing_decom:
    if i > -14:
        print("bigger than four")
    else:
        print("sikeeeee")
        


# In[6]:


import numpy as np
import scipy
from scipy import linalg

magnet_array = [[1, 1, 1, 1, 1],
               [1.2, 0.1, 5, 7, 8],
               [1, 0, 1.4, 1, 1]]
mag_transpose = np.transpose(magnet_array, axes = None)

magnet_array_identity = np.dot(magnet_array, mag_transpose)
print(magnet_array_identity)


eigen_stuff = scipy.linalg.eig(magnet_array_identity)
print(eigen_stuff)

for range in eigen_stuff:
    avg_eig = np.mean(range)
    print(avg_eig)
    


# In[12]:


import numpy as np
import scipy 
from scipy import linalg


vect_list = [1.231, .7, 1.2]
ordered_vecs = list((enumerate(vect_list)))

d = dict((i,j) for i, j in ordered_vecs)
print(d)

##now convert back to a list that can be 
##mapped to propositions


# In[15]:





# In[ ]:




