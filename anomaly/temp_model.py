import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data
from tsmoothie.smoother import *
import paho.mqtt.client as mqtt

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



## how to act when connecting to broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("enes/iot")
    #client.subscribe("cooler/temp")

## action to take when receiving message.
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))
    print("Received message " + str(msg.payload.decode("utf-8")))

client = mqtt.Client("enes")
client.on_connect = on_connect
client.on_message = on_message

client.connect("test.mosquitto.org",1883, 60)

client.loop_forever()