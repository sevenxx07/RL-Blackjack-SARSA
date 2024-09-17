import math

from abstractagent import AbstractAgent
from blackjack import BlackjackEnv, BlackjackObservation
from carddeck import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class SarsaAgent(AbstractAgent):
    """
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate) as long as you fulfil convergence criteria.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        super().__init__(env, number_of_episodes)
        self.alpha = 100  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.greedy = 0.9
        self.q_values = {}  # Dictionary to store Q-values
        self.visit_state = {}
        self.curr_state = np.zeros(number_of_episodes, dtype=float)
        self.q_values['13,6,False'] = np.zeros(self.env.action_space.n, dtype=float)

    def train(self):
        for i in range(self.number_of_episodes):
            #print("Episode:", i)
            observation, _ = self.env.reset()
            terminal = False
            prev = observation
            action = self.choose_action(prev, i)
            while not terminal:
                observation, reward, terminal, _, _ = self.env.step(action)
                next_action = self.choose_action(observation, i)
                self.update_q_values(prev, action, reward, observation, next_action)
                prev = observation
                action = next_action
            self.curr_state[i] = np.average(self.q_values['13,6,False'])
        print(str(self.q_values['16,11,True'][1]))
        print(str(self.q_values['16,11,True'][0]))
        plt.plot(self.curr_state)
        plt.savefig('currstate.png')


    def choose_action(self, observation: BlackjackObservation, epoch: int) -> int:
        """
        Chooses an action based on epsilon-greedy policy.
        """
        mygreedy = self.greedy*math.exp(-0.001*(epoch+1))

        if "ACE" in str(observation.player_hand):
            has_ace = True
        else:
            has_ace = False
        state = str(observation.player_hand.value()) + "," + str(observation.dealer_hand.value()) + "," + str(has_ace)
        if np.random.random() < mygreedy or state not in self.q_values:  # Exploration
            return self.env.action_space.sample()
        else:  # Exploitation
            return self.get_optimal_action(observation)

    def get_optimal_action(self, observation: BlackjackObservation) -> int:
        """
        Returns the action with the highest Q-value for the given observation.
        """
        if "ACE" in str(observation.player_hand):
            has_ace = True
        else:
            has_ace = False
        state = str(observation.player_hand.value()) + "," + str(observation.dealer_hand.value()) + "," + str(has_ace)

        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.env.action_space.n, dtype=float)
        return np.argmax(self.q_values[state])

    def update_q_values(self, prev: BlackjackObservation, action: int, reward: float,
                        observation: BlackjackObservation, next_action: int):
        """
        Updates the Q-values using the SARSA update rule.
        """
        if "ACE" in str(observation.player_hand):
            has_ace = True
        else:
            has_ace = False
        state = str(observation.player_hand.value())+","+ str(observation.dealer_hand.value())+","+ str(has_ace)
        if "ACE" in str(prev.player_hand):
            has_ace = True
        else:
            has_ace = False
        prevstate = str(prev.player_hand.value())+","+ str(prev.dealer_hand.value())+","+ str(has_ace)
        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.env.action_space.n, dtype=float)
        if prevstate not in self.q_values:
            self.q_values[prevstate] = np.zeros(self.env.action_space.n, dtype=float)
        if prevstate not in self.visit_state:
            self.visit_state[prevstate] = 0.0
        if state not in self.visit_state:
            self.visit_state[state] = 0.0
        td_target = reward + self.gamma * self.q_values[state][next_action]
        learning_rate = self.alpha/(self.alpha-0+self.visit_state[prevstate])
        self.visit_state[prevstate] += 1
        td_error = td_target - self.q_values[prevstate][action]
        self.q_values[prevstate][action] += learning_rate * td_error

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool, action: int) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned Q value for
        particular observation and action.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :param action: Action for Q-value.
        :return: The learned Q-value for the given observation and action.
        """
        if observation not in self.q_values:
            self.q_values[observation] = np.zeros(self.env.action_space.n, dtype=float)
        return self.q_values[observation][action]
