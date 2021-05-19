
from collections import OrderedDict, defaultdict
from typing import Dict, Union

from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from math import sqrt

Score = Union[int, float]

C, D = Action.C, Action.D


class RegularQLearner(Player):
    
    """
    This player learns the best strategies through the q-learning algorithm.
    
    - This is an adaptation of the Risky Q Learner that was created by Geraint Palmer. 
    - The Q-function and state-representation has been changed so that the agent acts self-interested.
    - There is also an option to add annealing epsilon to the algorithm. 
    - Instructions on which parameters to change to add annealing epsilon are written next to them.  
    
    - by Anna Dollbo 
    """

    name = "Regular QLearner"
    classifier = {
        "memory_depth": float("inf"),  # Long memory
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    
   

    gamma = 0.95 #discount rate: how much the algorithm cares about future reward. Close to 0 for short-term rewards, close to 1 for long-term rewards. 

   
    def __init__(self, memory_length=1, turns_in_game = 200, part_of_MMQL=False, name_id=None, alpha=0.5, epsilon=1.0) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True

        self.prev_action = None  # type: Action
        self.original_prev_action = None  # type: Action
        self.score = 0
        self.turns = 0 #count for turns
        self.Qs = {"": dict(zip([C, D], [0, 0]))}
        self.Vs = {"": 0}
        self.prev_state = ""
        self.rewards_list = []
        self.memory_length = memory_length
        self.turns_in_game = turns_in_game
        self.part_of_MMQL = part_of_MMQL
        self.name_id = name_id
        self.alpha = alpha 
        self.action = None
        self.previous_Vs = {"": 0}
        self.current_policy_time = 0
        self.list_size = 20 #if using MMQL, set same a MMQL's self.policy_eval_point 
        
        #epsilon parameters
        self.epsilon = epsilon #probability of choosing actions by random
        self.epsilon_decay = True # set to False if you don't want epsilon decay 
        self.start_epsilon_decay = self.epsilon #should be same as epsilon 
        self.end_epsilon_decay = self.turns_in_game//1.5 #the last 1/3 of the game the actions will be deterministic
        self.decay_value = self.epsilon/(self.end_epsilon_decay - self.start_epsilon_decay)     
        
    
    def __repr__(self):
        return "Regular Q-Learner"
        

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

   
     def strategy(self, opponent: Player) -> Action:
        """Runs a qlearning algorithm while the tournament is running."""
        if len(self.history) == 0:
            self.prev_action = self._random.random_choice()
            self.original_prev_action = self.prev_action
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        if state not in self.Qs:
            self.Qs[state] = dict(zip([C, D], [0, 0]))
            self.Vs[state] = 0
        self.perform_q_learning(self.prev_state, state, self.prev_action, reward)
        self.action = self.select_action(state)
        self.prev_state = state
        self.prev_action = self.action
        if self.part_of_MMQL == True:
            pass
        else:
            return self.action
        
    
    def select_action(self, state: str) -> Action:
        """
        Selects the action based on the epsilon-greedy policy. 
        If using annealing epsilon it decays epsilon before returning action.
        """
        rnd_num = self._random.random()
        p = 1.0 - self.epsilon
        if rnd_num > p:
            action = self._random.random_choice() 
        else:
            action = max(self.Qs[state], key=lambda x: self.Qs[state][x])
        if self.epsilon_decay == True:
            self.turns += 1
            if self.turns < self.end_epsilon_decay:
                self.epsilon -= self.decay_value      
        return action 

    
    def find_state(self, opponent: Player) -> str:
        """
        Finds the state (the opponents last n moves).
        """
        action_str = actions_to_str(opponent.history[-self.memory_length :])
        return action_str 

    
    def perform_q_learning(self, prev_state: str, state: str, action: Action, reward):
        """
        Performs the qlearning algorithm.
        Updates Q- and V-values.
        """
        max_future_q = max(self.Qs[state].values())
        self.Qs[prev_state][action] = (1.0 - self.alpha) * self.Qs[prev_state][
            action] + self.alpha * (reward + self.gamma * max_future_q)
        self.Vs[prev_state] = max(self.Qs[prev_state].values())

        
        
    def find_reward(self, opponent: Player) -> Dict[Action, Dict[Action, Score]]:
        """
        Finds the reward gained on the last iteration.
        """

        if len(opponent.history) == 0:
            opp_prev_action = self._random.random_choice()
        else:
            opp_prev_action = opponent.history[-1]
        reward = self.payoff_matrix[self.prev_action][opp_prev_action]
        if self.part_of_MMQL:
            self.rewards_list_manager(reward)
        return reward
    
   
    def rewards_list_manager(self, reward):
        """
        Appends the last reward to the rewards list and makes sure it only holds the last n items.
        """
        self.rewards_list.append(reward)
        if len(self.rewards_list) > self.list_size:
            self.rewards_list.pop(0)
            
            
    
class MixedMemoryQLearner(Player):
    
    """
    A meta-strategy algorithm that uses 3 Regular Q-learner players with different memory lengths. 
    This strategy changes between the strategy of the 3 different Qlearning agents during the game based on reward-evaluation.
    This algorithm remembers if it has played an opponent before and loads the previous best play at beginning of game.
    
    - Created by Anna Dollbo
    """
    
    name = "Mixed Memory Q-Learner"
    classifier = {
        "memory_depth": float("inf"),  
        "stochastic": True,
        "long_run_time": True, 
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    

    def __init__(self, previous_matches_dict=None, Vs_dict=None, game_length=200, learning_rate=0.9, epsilon_val=1.0, convergence_check=False, Vval_list=None, game_info_dict=None):
   
        self.classifier["stochastic"] = True

        self.score = 0
        self.rounds = 0
        self.turns = 0
        self.current_policy = None 
        self.previous_matches_dict = previous_matches_dict
        self.Vs_dict = Vs_dict
        self.game_length = game_length  
        self.learning_rate = learning_rate
        self.epsilon_val = epsilon_val
        self.convergence_check = convergence_check
        self.Vval_list = Vval_list
        self.convergence_threshold = 0.001
        self.Vs_eval_point = 5
        self.policy_eval_point = 20
        self.game_info_dict = game_info_dict
        
        #epsilon parameters
        self.epsilon = 1.0 
        self.start_epsilon_decay = 1.0 #should be same as epsilon 
        self.end_epsilon_decay = self.game_length
        self.decay_value = self.epsilon/(self.end_epsilon_decay - self.start_epsilon_decay)
        
        #Player instances 
        self.short_mem_player = RegularQLearner(memory_length=1, part_of_MMQL=True, turns_in_game=self.game_length, name_id="short", alpha=self.learning_rate, epsilon=self.epsilon_val)
        self.mid_mem_player = RegularQLearner(memory_length=5, part_of_MMQL=True, turns_in_game=self.game_length, name_id="mid", alpha=self.learning_rate, epsilon=self.epsilon_val)
        self.long_mem_player = RegularQLearner(memory_length=10, part_of_MMQL=True, turns_in_game=self.game_length, name_id="long", alpha=self.learning_rate, epsilon=self.epsilon_val)
        
        #List of policies
        self.policy_list = [self.short_mem_player, self.mid_mem_player, self.long_mem_player]
        # self.short_mem_player, self.mid_mem_player, self.long_mem_player
        
        super().__init__()
        
        
    def __repr__(self):
        return "Mixed Memory Q-Learner"
    
        
    def set_seed(self, seed=None):
        """
        Seeds the players in policy_list.
        """
        super().set_seed(seed=seed)
        for t in self.policy_list:
            t.set_seed(self._random.random_seed_int())
            
    
    def strategy(self, opponent: Player) -> Action:
        """
        Sends opponent to subplayers, chooses action from policy that performs best.
        """
        if self.turns == 0:
            print(f"Now playing {opponent.name}.")
            self.load_policies(opponent)
            self.set_current_policy(opponent)
        for policy in self.policy_list:
            policy.strategy(opponent)
        if self.evaluation_point():
            self.evaluate_policies()
        action = self.current_policy.action 
        if self.convergence_check == True:
            if self.vf_eval_point():
                self.evaluate_valfunc(opponent)
        self.turns += 1
        self.curr_policy_time_calc()
        if self.turns < self.end_epsilon_decay:
            self.epsilon -= self.decay_value
        if self.turns == self.game_length:
            print("End of game")
            self.unload_current_policy(opponent)
            self.unload_currpolicy_percentage(opponent)
        return action
    
            
    def has_met_before(self, opponent: Player):
        """
        Checks in its history if it has played this Player before.
        """
        if opponent.name in self.previous_matches_dict:
            return True
            
            
    def load_policies(self, opponent: Player):
        """
        Loads previous Qvalues for each player from the last time they played opponent.
        """
        if self.has_met_before(opponent):
            self.rounds = self.get_value(self.previous_matches_dict[opponent.name], 'round')
            for policy in self.policy_list:
                policy.Qs = self.get_value(self.previous_matches_dict[opponent.name], policy.name_id)
                policy.Vs = self.get_value(self.Vs_dict[opponent.name], policy.name_id)

            
    def set_current_policy(self, opponent: Player):
        """
        Sets current policy. Either randomly if it hasn't met player before. Otherwise previous best.
        """
        if not self.has_met_before(opponent):
            self.current_policy = self._random.choice(self.policy_list)
        else:
            for policy in self.policy_list:
                if self.get_value(self.previous_matches_dict[opponent.name], 'best') == policy.name_id:
                    self.current_policy = policy
        print(f"Starting with memory length {self.current_policy.memory_length}.")
        
   
    def get_value(self, listOfDicts, key):
        """
        Returns the value of a key of a dictionary that is in a list.
        """
        for val in listOfDicts:
            if key in val:
                return val[key]
       
                    
    def evaluation_point(self):
        """
        Checks whether this time-step is an evaluation point or not.
        """
        if self.turns in range(40,(self.game_length - 19), self.policy_eval_point):
                return True
            
    
    def evaluate_policies(self):
        """
        Evaluate policies and switches current_policy if mean average reward for the last n steps > than current policy.
        """
        combined_rewards = {}
        previous_current_policy = self.current_policy
        for policy in self.policy_list:
            mean_reward_f = lambda x: (sum(x)/len(x))
            value = mean_reward_f(policy.rewards_list)
            #print(f" Memory length: {policy.memory_length}. Mean reward: {value}.")
            combined_rewards[policy.name_id] = value
        best_policy = self.find_max_value(combined_rewards)
        if self.current_policy.name_id in best_policy:
            print(f" Still using memory length {self.current_policy.memory_length}. Turn: {self.turns}.")  
        else:
            self.policy_switching(best_policy)
        if self.current_policy != previous_current_policy:
            print(f"Switched to memory length {self.current_policy.memory_length}. Turn: {self.turns}.")
            

    def find_max_value(self, dictionary):
        """
        Finds the highest value in a dictionary and returns a list with the keys with that value.
        """
        max_value = max(dictionary.items(), key=lambda x: x[1])
        list_of_max_values = []
        for k, v in dictionary.items():
            if v == max_value[1]:
                list_of_max_values.append(k)
        return list_of_max_values

            
            
    def policy_switching(self, listObject):
        """
        Switches current policy if there is policy with higher reward over last n turns. 
        """
        rnd_num = self._random.random()
        p = 1.0 - self.epsilon
        if rnd_num > p:
            for policy in self.policy_list:
                if len(listObject) == 1 and policy.name in listObject:
                    self.current_policy = policy
                else:
                    choice = self._random.choice(listObject)
                    if policy.name_id == choice:
                        self.current_policy = policy
        else:
            pass
        
        
    def curr_policy_time_calc(self):
        """
        Counts how many turns each policy is played as current policy.
        """
        for policy in self.policy_list:
            if policy == self.current_policy:
                policy.current_policy_time += 1
                
                    
    def unload_currpolicy_percentage(self, opponent: Player):
        """
        Unloads how many percent of each round each player was current policy into global dict.
        """
        empty_dict = {}
        for policy in self.policy_list:
            percentage = int((policy.current_policy_time/self.game_length)*100)
            empty_dict[policy.name_id] = percentage  
        self.game_info_dict[opponent.name][self.rounds].append(empty_dict)
        
                       
    def unload_current_policy(self, opponent: Player):
        """
        Unloads the Qvalues for each policy at the end of the game into global dict.
        """
        Qs_list = []
        Vs_list = []
        qs_dict = {}
        vs_dict = {}
        self.rounds += 1
        for policy in self.policy_list:
            qs_dict[policy.name_id] = policy.Qs
            vs_dict[policy.name_id] = policy.Vs
            if policy == self.current_policy:
                qs_dict['best'] = policy.name_id
                qs_dict['round'] = self.rounds
        Qs_list.append(qs_dict)
        Vs_list.append(vs_dict)
        if self.has_met_before(opponent):
            self.previous_matches_dict.pop(opponent.name)
            self.Vs_dict.pop(opponent.name)
        self.previous_matches_dict[opponent.name] = Qs_list
        self.Vs_dict[opponent.name] = Vs_list


    def vf_eval_point(self): 
        """
        Checks if it is time to evaluate policy. 
        """
        if self.turns in range(0,(self.game_length + 1),self.Vs_eval_point):
            return True
    
    
    def evaluate_valfunc(self, opponent: Player):
        """
        Method that calculates the RSME for each policy over every N step. Used to check convergence. 
        Optional.
        """
        for policy in self.policy_list:
            summation = 0
            if self.turns == 0:
                policy.previous_Vs = dict(policy.Vs)
                break
            for state in policy.Vs:
                if state not in policy.previous_Vs:
                    policy.previous_Vs[state] = 0
            if len(policy.previous_Vs) != len(policy.Vs):
                print("These dictionaries do not match!")         
            for key1 in policy.Vs:
                for key2 in policy.previous_Vs:
                    if key1 == key2:
                        delta = policy.Vs[key1] - policy.previous_Vs[key2]
                        delta_squared = delta**2
                        summation += delta_squared
            n = len(policy.Vs)
            MSE = summation/n
            RMSE = sqrt(MSE)
            self.Vval_list[policy.name_id][opponent.name].append(RMSE)
            #if MSE < self.convergence_threshold:
                #print(f"Converged at step {self.total_steps()} for memory length {policy.memory_length} against {opponent.name}.")
            policy.previous_Vs = dict(policy.Vs)

            
    def total_steps(self):
        """
        Checks how many turns over the whole tournament has passed at the timepoint. 
        """
        return self.turns + (self.rounds*self.game_length)
        
        
            
            
            
            
