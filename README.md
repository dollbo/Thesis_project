# Mixed Memory Q-Learner Thesis Project

This repository contains files from my thesis project "Mixed Memory Q-Learner - an adaptive reinforcement learning algorithm for the Iterated Prisoner's Dilemma". 

MMQL is a meta-strategy using a team of Q-learning players with different state-representations (in this context: memory length) to play the IPD. MMQL has the ability to recognise its opponents between games and can continue its learning throughout interactions with the same opponent. MMQL evaluates which memory length gives highest reward during run time and switches which player its using to play the opponent based on this evaluation.  

MMQL was built to fit within the Axelrod Python Library.
The strategy has not yet been committed to the library.


