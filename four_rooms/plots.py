import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import deepdish as dd



#####################################################################################

def plot1():
    n = np.linspace(1,50,50)
    everything = 2**(2**n)
    fact = np.array([np.math.factorial(x) for x in n])
    OR = 2**n - 1
    standard = n
    
    s = 20
    rc_ = {'figure.figsize':(11,8), 'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig,ax=plt.subplots()
    plt.plot(everything, linewidth=5.0, label="Boolean task algebra")
    plt.plot(OR, linewidth=5.0, label="Disjunction only")
    plt.plot(standard, linewidth=5.0, label="No transfer")
    #plt.plot(fact, '--', label="reference, n!")
    plt.yscale('log', basey=10)
    plt.xlim(1, 10)
    plt.ylim(1, 10**18)
    plt.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Number of solvable tasks')
    plt.show()
    fig.savefig("plots/analytic.pdf", bbox_inches='tight')
#####################################################################################

def plot2():
    data1 = dd.io.load('exps_data/exp1_samples_Qs.h5')
    data2 = dd.io.load('exps_data/exp1_samples_EQs.h5')
    
    mean1 = np.cumsum(data1.mean(axis=0))
    std1 = data1.std(axis=0)
    mean2 = np.cumsum(data2.mean(axis=0))
    std2 = data2.std(axis=0)
    
    s = 20
    rc_ = {'figure.figsize':(11,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig,ax=plt.subplots()
    ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label=r"Extended $Q$-function")
    ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label=r"$Q$-function")
    plt.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to converge')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    plt.show()
    fig.savefig("plots/cum_bar.pdf", bbox_inches='tight')
# #####################################################################################

def plot3():
    data1 = dd.io.load('exps_data/exp2_samples_Qs.h5')
    data2 = dd.io.load('exps_data/exp2_samples_EQs.h5')
    
    n = 50
    x = np.arange(1,n+1)
    mean1 = np.cumsum(data1.mean(axis=0))
    mean1 = np.array(list(mean1)+[mean1[-1]]*(n-len(mean1)))
    std1 = data1.std(axis=0)
    std1 = np.array(list(std1)+[std1[-1]]*(n-len(std1)))
    mean2 = np.cumsum(data2.mean(axis=0))
    mean2 = np.array(list(mean2)+[mean2[-1]]*(n-len(mean2)))
    std2 = data2.std(axis=0)
    std2 = np.array(list(std2)+[std2[-1]]*(n-len(std2)))
    
    width = 0.5  # the width of the bars
    s = 20
    rc_ = {'figure.figsize':(11,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, mean2, width, yerr=std2, align='center', ecolor='black', label="Boolean task algebra")
    ax.bar(x + width/2, mean1, width, yerr=std1, align='center', ecolor='black', label="Disjunction only")
    ax.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to converge')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    plt.show()
    fig.savefig("plots/40goals_cum_bar.pdf", bbox_inches='tight')

#####################################################################################

def plot4():
    tasks = [r'${M_{\emptyset}}$',
              r'${M_{\mathcal{U}}}$',
              r'${M_{T}}\wedge{M_{L}}$',
              r'${M_{T}}\wedge\neg{M_{L}}$',
              r'${M_{L}}\wedge\neg{M_{T}}$',
              r'${M_{T}}\bar{\vee}{M_{L}}$',
              r'${M_{T}}$',
              r'$\neg {M_{T}}$',
              r'${M_{L}}$',
              r'$\neg {M_{L}}$',
              r'${M_{T}}\vee{M_{L}}$',
              r'${M_{T}}\vee\neg{M_{L}}$',
              r'${M_{L}}\vee\neg{M_{T}}$',
              r'${M_{T}}\bar{\wedge}{M_{L}}$',
              r'$\neg({M_{T}} \veebar {M_{L}})$',
              r'${M_{T}} \veebar {M_{L}}$'
              ]
    
    plt.ylim(-0.5, 2)
    rc_ = {'figure.figsize':(30,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
    
    n = 2
    
    #data0 = dd.io.load('exps_data/trash/exp3_returns_optimal.h5')
    data0 = dd.io.load('exps_data/exp3_returns_0.h5')/10
    data1 = dd.io.load('exps_data/exp3_returns_2.h5')/10
    data2 = dd.io.load('exps_data/exp3_returns_1.h5')/10
    data3 = dd.io.load('exps_data/exp3_returns_3.h5')/10
    
    types = ["Sparse rewards and Same absorbing set",
              "Dense rewards and Same absorbing set",
              "Sparse rewards and Different absorbing set",
              "Dense rewards and Different absorbing set",
            ]
    
    data = pd.DataFrame(
    [[data0[i,t] for t in range(n,16)]+[types[0]] for i in range(len(data1))] +
    [[data1[i,t] for t in range(n,16)]+[types[1]] for i in range(len(data1))] +
    [[data2[i,t] for t in range(n,16)]+[types[2]] for i in range(len(data1))] +
    [[data3[i,t] for t in range(n,16)]+[types[3]] for i in range(len(data1))],
      columns=tasks[n:]+["Domain"])
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3, showfliers = False)
    plt.show()
    fig.savefig("plots/dense.pdf", bbox_inches='tight')


#####################################################################################

def plot5():
    tasks = [r'${M_{\emptyset}}$',
              r'${M_{\mathcal{U}}}$',
              r'${M_{T}}\wedge{M_{L}}$',
              r'${M_{T}}\wedge\neg{M_{L}}$',
              r'${M_{L}}\wedge\neg{M_{T}}$',
              r'${M_{T}}\bar{\vee}{M_{L}}$',
              r'${M_{T}}$',
              r'$\neg {M_{T}}$',
              r'${M_{L}}$',
              r'$\neg {M_{L}}$',
              r'${M_{T}}\vee{M_{L}}$',
              r'${M_{T}}\vee\neg{M_{L}}$',
              r'${M_{L}}\vee\neg{M_{T}}$',
              r'${M_{T}}\bar{\wedge}{M_{L}}$',
              r'$\neg({M_{T}} \veebar {M_{L}})$',
              r'${M_{T}} \veebar {M_{L}}$'
              ]
        
    s = 20
    rc_ = {'figure.figsize':(30,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
    
    n = 2
    
    for i in range(4):
        data0 = dd.io.load('exps_data/exp5_returns_'+str(i)+'.h5')[:1000,:]
        data1 = dd.io.load('exps_data/exp4_returns_'+str(i)+'.h5')[:1000,:]
        
        types = ["Optimal",
                  "Composed",
                ]
        
        data = pd.DataFrame(
        [[data0[i,t] for t in range(n,16)]+[types[0]] for i in range(len(data1))] +
        [[data1[i,t] for t in range(n,16)]+[types[1]] for i in range(len(data1))],
          columns=tasks[n:]+[""])
        data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
        
        fig, ax = plt.subplots()
        ax = sns.boxplot(x="Tasks", y="Average Returns", hue="", data=data, linewidth=3, showfliers = False)
        plt.show()
        fig.savefig("plots/dense_sp_"+str(i)+".pdf", bbox_inches='tight')


plot1();plot2();plot3();plot4();plot5()
