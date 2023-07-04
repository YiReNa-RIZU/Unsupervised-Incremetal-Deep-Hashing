#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:00:07 2021

@author: ye
"""

import matplotlib.pyplot as plt
import os


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist))

    y1 = hist


    plt.plot(x, y1, label='loss')


    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()
    
    
def mAP_plot(hist1, hist2, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist2))

    y1 = hist1
    y2 = hist2


    plt.plot(x, y1, label='mAP_up')
    plt.plot(x, y2, label='mAP_down')


    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_map.png')

    plt.savefig(path)

    plt.close()