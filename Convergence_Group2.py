#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:00:28 2021

@author: anna
"""

from collections import defaultdict
import pprint
import axelrod as axl
import random
import pandas as pd
import numpy as np 

"""
MixedMemoryQLearner against 9 default strategies
alpha 0.9, epsilon 1.0, decay//1.5
"""
random.seed(1)
savedQs_MMQL10 = {} 
Vs = {}
Delta_Vvalues = defaultdict(lambda: defaultdict(list))
Game_Info = defaultdict(lambda: defaultdict(list))


strategies = [axl.Cooperator(), axl.Defector(),
              axl.TitForTat(), axl.Grudger(),
              axl.Random(), axl.TwoTitsForTat(), 
              axl.SuspiciousTitForTat(), axl.TitFor2Tats(), 
              axl.WinStayLoseShift(), 
              axl.MixedMemoryQLearner(previous_matches_dict=savedQs_MMQL10, Vs_dict=Vs, game_length=200, 
                                      learning_rate=0.9, epsilon_val=1.0, Vval_list=Delta_Vvalues,
                                      game_info_dict=Game_Info, convergence_check=True)]


tournament = axl.Tournament(strategies, turns=200, repetitions=5, match_attributes={"length": float('inf')})
results = tournament.play()

print(results.ranked_names)
plot = axl.Plot(results)
plot.boxplot();
print(results.scores)
print("Total score: " + str(sum(results.scores[9])))
print("Normalised scores per opponent and turns:\n" + str(["%.3f" % item for item in results.normalised_scores[9]])) 
print("Mean payoff against each opponent: \n" + str(["%.3f" % item for item in results.payoff_matrix[9]]) )
print(results.summarise())
results.write_summary('MMQL_Summary_a09_e01_5reps_RMSE.csv')
print(Delta_Vvalues)

df = pd.DataFrame.from_dict(Delta_Vvalues)
df.to_csv(r"/home/anna/Thesis_Project/MMQL_Convals_a09_e01_5reps_RMSE.csv")

#df2 = pd.DataFrame.from_dict(Game_Info, orient="columns", dtype = np.int32)
#df.to_csv(r"/home/anna/Thesis_Project/Gamepercentage_a09_e1_5reps.csv")

print(Game_Info)
df2 = pd.DataFrame.from_dict(Game_Info, orient="index").stack().to_frame()
#df2_1 = pd.DataFrame(df[0].values.tolist(), index=df.index)
df2.to_csv(r"/home/anna/Thesis_Project/Gamepercentage_a09_e01_5reps.csv")



