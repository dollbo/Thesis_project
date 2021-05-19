
"""
Created on Thu May  6 10:14:30 2021

@author: anna
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval
import scipy 



df = pd.read_csv('Reg10CONT_convals_a05_e01_5reps_RMSE.csv')


""" Memory 1 convergence all opponents"""
mem1_val1 = literal_eval(df.iloc[0,1])
mem1_val2 = literal_eval(df.iloc[1,1])
mem1_val3 = literal_eval(df.iloc[2,1])
v1= literal_eval(df.iloc[3,1])
mem1_val4 = v1[0:195]
mem1_val5 = literal_eval(df.iloc[4,1])
mem1_val6 = literal_eval(df.iloc[5,1])
mem1_val7 = literal_eval(df.iloc[6,1])
mem1_val8 = literal_eval(df.iloc[7,1])
mem1_val9 = literal_eval(df.iloc[8,1])
v = literal_eval(df.iloc[9,1])
mem1_val10 = v[0:195]

"""
# Memory 5 convergence all opponents
mem5_val1 = literal_eval(df.iloc[0,2])
mem5_val2 = literal_eval(df.iloc[1,2])
mem5_val3 = literal_eval(df.iloc[2,2])
mem5_val4 = literal_eval(df.iloc[3,2])
mem5_val5 = literal_eval(df.iloc[4,2])
mem5_val6 = literal_eval(df.iloc[5,2])
mem5_val7 = literal_eval(df.iloc[6,2])
mem5_val8 = literal_eval(df.iloc[7,2])
mem5_val9 = literal_eval(df.iloc[8,2])
v = literal_eval(df.iloc[9,2])
mem5_val10 = v[0:195]


# Memory 10 convergence all opponents
mem10_val1 = literal_eval(df.iloc[0,3])
mem10_val2 = literal_eval(df.iloc[1,3])
mem10_val3 = literal_eval(df.iloc[2,3])
mem10_val4 = literal_eval(df.iloc[3,3])
mem10_val5 = literal_eval(df.iloc[4,3])
mem10_val6 = literal_eval(df.iloc[5,3])
mem10_val7 = literal_eval(df.iloc[6,3])
mem10_val8 = literal_eval(df.iloc[7,3])
mem10_val9 = literal_eval(df.iloc[8,3])
v = literal_eval(df.iloc[9,3])
mem10_val10 = v[0:195]

"""

plt.plot(mem1_val1, label = "Cooperator", color = "blue")
plt.plot(mem1_val2, label = "Defector", color = "red")

#"""

plt.plot(mem1_val3, label = "TFT", color = "green")
plt.plot(mem1_val6, label = "2TFT", color = "gold")
plt.plot(mem1_val7, label = "Suspicious TFT", color = "dimgrey")
plt.plot(mem1_val8, label = "TF2T", color = "darkturquoise")
#"""
#"""
plt.plot(mem1_val9, label = "WinStayLoseShift", color = "sienna")
plt.plot(mem1_val4, label = "Grudger", color = "darkviolet")
plt.plot(mem1_val5, label = "Random", color = "magenta")
#"""

plt.plot(mem1_val10, label = "MMQL", color = "lawngreen")
plt.xlabel("No. evaluation points over tournament")
plt.ylabel("Root Mean Squared Error")
plt.ylim(0, 10)
#plt.xlim(0,50)
#plt.xlim(150,195)
plt.legend(loc = "upper right")
plt.title("RMSE for Reg cont. Mem10")
plt.show()




