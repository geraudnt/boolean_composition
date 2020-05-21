import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import deepdish as dd



#####################################################################################

n = np.linspace(1,50,50)
everything = 2**(2**n)
fact = np.array([np.math.factorial(x) for x in n])
OR = 2**n - 1
standard = n

plt.figure(0)
params = {'legend.fontsize': 30,'font.size': 40}
plt.rcParams.update(params)
plt.plot(everything, linewidth=10.0, label="Boolean task algebra")
plt.plot(OR, linewidth=10.0, label="Disjunction only")
plt.plot(standard, linewidth=10.0, label="No transfer")
#plt.plot(fact, '--', label="reference, n!")
plt.yscale('log', basey=10)
plt.xlim(1, 10)
plt.ylim(1, 10**18)
plt.legend()
#plt.xlabel("Learned tasks (n)", fontsize=40)
#plt.ylabel("Zero-shot solvable Tasks", fontsize=40)
plt.show()

#####################################################################################
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

fig, ax = plt.subplots()
width = 0.5  # the width of the bars
params = {'legend.fontsize': 20}
plt.rcParams.update(params)
ax.bar(x - width/2, mean2, width, yerr=std2, align='center', color='b', ecolor='g', label="Boolean task algebra")
ax.bar(x + width/2, mean1, width, yerr=std1, align='center', color='tab:orange', ecolor='g', label="Disjunction only")
ax.legend()
#ax.set_xlabel("Tasks", fontsize=20)
#ax.set_ylabel('Cumulative steps to converge', fontsize=20)
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
#ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
fig.tight_layout()
plt.show()

#####################################################################################

# EQ_min,EQ_max,AND(A,B),AND(A,NEG(B)),AND(B,NEG(A)),NEG(OR(A,B)),A,NEG(A),B,NEG(B),OR(A,B),OR(A,NEG(B)),OR(B,NEG(A)),NEG(AND(A,B)),NEG(XOR(A,B)),XOR(A,B)

tasks = [r'$\bar{Q}^{*}_{M_{\emptyset}}$',
         r'$\bar{Q}^{*}_{M_{\mathcal{U}}}$',
         r'$\bar{Q}^{*}_{M_{T}}\wedge\bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{T}}\wedge\neg\bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{L}}\wedge\neg\bar{Q}^{*}_{M_{T}}$',
         r'$\bar{Q}^{*}_{M_{T}}\bar{\vee}\bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{T}}$',
         r'$\neg \bar{Q}^{*}_{M_{T}}$',
         r'$\bar{Q}^{*}_{M_{L}}$',
         r'$\neg \bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{T}}\vee\bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{T}}\vee\neg\bar{Q}^{*}_{M_{L}}$',
         r'$\bar{Q}^{*}_{M_{L}}\vee\neg\bar{Q}^{*}_{M_{T}}$',
         r'$\bar{Q}^{*}_{M_{T}}\bar{\wedge}\bar{Q}^{*}_{M_{L}}$',
         r'$\neg(\bar{Q}^{*}_{M_{T}}\veebar\bar{Q}^{*}_{M_{L}}$)',
         r'$\bar{Q}^{*}_{M_{T}}\veebar\bar{Q}^{*}_{M_{L}}$'
         ]

plt.ylim(-0.5, 2)
rc = {'figure.figsize':(30,10),'axes.labelsize': 30, 'font.size': 30, 
      'legend.fontsize': 20, 'axes.titlesize': 30}
sns.set(rc=rc, style="darkgrid",font_scale = 1.8)
n = 2

#data0 = dd.io.load('exps_data/trash/exp3_returns_optimal.h5')
data0 = dd.io.load('exps_data/exp3_returns_0.h5')
data1 = dd.io.load('exps_data/exp3_returns_2.h5')
data2 = dd.io.load('exps_data/exp3_returns_1.h5')
data3 = dd.io.load('exps_data/exp3_returns_3.h5')

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

ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3)
plt.show()


