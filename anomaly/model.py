#https://towardsdatascience.com/real-time-time-series-anomaly-detection-981cf1e1ca13
#https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/Anomaly_Detection_RealTime/Anomaly_Detection_RealTime.ipynb

import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data
from tsmoothie.smoother import *

### UTILITY FUNCTION FOR PLOTTING ###

def plot_history(ax, i, is_anomaly, window_len, color='blue', **pltargs):

    posrange = np.arange(0,i)

    ax.fill_between(posrange[window_len:],
                    pltargs['low'][1:], pltargs['up'][1:],
                    color=color, alpha=0.2)
    if is_anomaly:
        ax.scatter(i-1, pltargs['original'][-1], c='red')
    else:
        ax.scatter(i-1, pltargs['original'][-1], c='black')
    ax.scatter(i-1, pltargs['smooth'][-1], c=color)

    ax.plot(posrange, pltargs['original'][1:], '.k')
    ax.plot(posrange[window_len:],
            pltargs['smooth'][1:], color=color, linewidth=3)

    if 'ano_id' in pltargs.keys():
        if pltargs['ano_id'].sum()>0:
            not_zeros = pltargs['ano_id'][pltargs['ano_id']!=0] -1
            ax.scatter(not_zeros, pltargs['original'][1:][not_zeros],
                       c='red', alpha=1.)

##########                  POINT ANOMALIES : RANDOM WALKS ########################
## GENERATE DATA ###

np.random.seed(42)

n_series, timesteps = 1, 200

data = sim_randomwalk(n_series=n_series, timesteps=timesteps,
                      process_noise=10, measure_noise=30)

# print(data.shape)
# print(type(data))

# plt.plot(data.T)
# np.set_printoptions(False)
# plt.show()


### SLIDING WINDOW PARAMETER ###
window_len = 20


### SIMULATE PROCESS REAL-TIME AND CREATE GIF ###
fig = plt.figure(figsize=(18,10))
camera = Camera(fig)

axes = [plt.subplot(n_series,1,ax+1) for ax in range(n_series)]
series = defaultdict(partial(np.ndarray, shape=(n_series,1), dtype='float32'))
anomaly_counter = 0
anomalies = []
for i in tqdm(range(timesteps+1), total=(timesteps+1)):
    #print(i)
    if i>window_len:

        smoother = ConvolutionSmoother(window_len=window_len, window_type='ones')
        smoother.smooth(series['original'][:,-window_len:])

        series['smooth'] = np.hstack([series['smooth'], smoother.smooth_data[:,[-1]]])

        _low, _up = smoother.get_intervals('sigma_interval', n_sigma=2)
        series['low'] = np.hstack([series['low'], _low[:,[-1]]])
        series['up'] = np.hstack([series['up'], _up[:,[-1]]])

        is_anomaly = np.logical_or(
            series['original'][:,-1] > series['up'][:,-1],
            series['original'][:,-1] < series['low'][:,-1]
        ).reshape(-1,1)
        if is_anomaly[0][0] == True :
             anomaly_counter += 1
        anomalies.append(is_anomaly[0][0])
        if is_anomaly.any():
            series['ano_id'] = np.hstack([series['ano_id'], is_anomaly*i]).astype(int)

        for s in range(n_series):
            pltargs = {k:v[s,:] for k,v in series.items()}
            plot_history(axes[s], i, is_anomaly[s], window_len,
                         **pltargs)

        camera.snap()

    if i>=timesteps:
        continue

    series['original'] = np.hstack([series['original'], data[:,[i]]])


print(anomaly_counter)
#print(len(series['original'][0]))

# print('CREATING GIF...')  # it may take a few seconds
# camera._photos = [camera._photos[-1]] + camera._photos
# animation = camera.animate()
# animation.save('animation1.gif')
# plt.close(fig)
# print('DONE')


### PLOT FINAL RESULT ###

# fig = plt.figure(figsize=(18,10))
# axes = [plt.subplot(n_series,1,ax+1) for ax in range(n_series)]

# for i,ax in enumerate(axes):

#     posrange = np.arange(window_len,timesteps)

#     ax.plot(series['original'][i,1:], '.k')
#     ax.plot(posrange, series['smooth'][i,1:], c='blue', linewidth=3)

#     ax.fill_between(posrange,
#                     series['low'][i,1:], series['up'][i,1:],
#                     color='blue', alpha=0.2)

#     ano_id = series['ano_id'][i][series['ano_id'][i] != 0] -1

#     if len(ano_id)>0:
#         ax.scatter(ano_id, series['original'][i,1:][ano_id],
#                    c='red', alpha=1.)

