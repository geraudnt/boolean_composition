import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import deepdish as dd


def plot1():
    tasks = ['Purple','Blue','Square','OR', 'AND', 'XOR']
        
    s = 20
    rc_ = {'figure.figsize':(10,5),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
        
    data0 = dd.io.load('data/exp_returns_0.h5')[:1000,:]
    data1 = dd.io.load('data/exp_returns_1.h5')[:1000,:]
    types = ["Optimal",
              "Composed",
            ]
    
    data = pd.DataFrame(
    [[data0[i,t] for t in range(len(tasks))]+[types[0]] for i in range(len(data1))] +
    [[data1[i,t] for t in range(len(tasks))]+[types[1]] for i in range(len(data1))],
      columns=tasks+[""])
    data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="", data=data, linewidth=3, showfliers = False)
    plt.show()
    fig.savefig("plots/returns.pdf", bbox_inches='tight')

#####################################################################################

def plot2():
    tasks = ['purple','blue','square','or', 'and', 'xor']
        
    s = 20
    rc_ = {'figure.figsize':(4,5),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
        
    data0 = dd.io.load('data/exp_returns_0.h5')[:1000,:]
    data1 = dd.io.load('data/exp_returns_1.h5')[:1000,:]
    types = ["Optimal",
              "Composed",
            ]
    
    for task in range(len(tasks)):
        data = data0[:,:2]
        data[:,0] = data0[:,task]
        data[:,1] = data1[:,task]
        data = pd.DataFrame(
        [[data[i,t] for t in range(len(data[i]))] for i in range(len(data))],
          columns=types)
        # data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
        
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=data, linewidth=3, showfliers = False)
        plt.xlabel('Tasks')
        plt.ylabel('Average Returns')
        # plt.show()
        fig.savefig("plots/returns_{0}.pdf".format(tasks[task]), bbox_inches='tight')

plot1();
plot2()