
"""
Created on Thu May  6 11:13:16 2021

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
alpha 0.5, epsilon 0.1, decay//1.5
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
                                      learning_rate=0.5, epsilon_val=0.1, Vval_list=Delta_Vvalues, 
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
#results.write_summary('Reg10CONT_summary_a05_e01_5reps_RMSE.csv')


df = pd.DataFrame.from_dict(Delta_Vvalues)
df.to_csv(r"/home/anna/Thesis_Project/Reg10CONT_convals_a05_e01_5reps_RMSE.csv", )

"""
df2 = pd.DataFrame.from_dict(Game_Info, orient="index").stack().to_frame()
#df2_1 = pd.DataFrame(df[0].values.tolist(), index=df.index)
df2.to_csv(r"/home/anna/Thesis_Project/Reg1CONTGamepercentage_a05_e01_5reps.csv")

"""




