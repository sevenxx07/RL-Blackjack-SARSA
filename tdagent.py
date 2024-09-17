from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from carddeck import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function
    and the get_hypothesis() method that returns the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal difference method.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        super().__init__(env, number_of_episodes)
        self.alpha = 100
        self.gamma = 0.95  # Discount factor
        self.state_action_values = {}  # Dictionary to store Q-values
        self.state_visits = {}
        self.curr_state = np.zeros(number_of_episodes)
        self.state_action_values['13,6,False'] = 0

    def train(self):
        for i in range(self.number_of_episodes):
            observation, _ = self.env.reset()
            terminal = False
            prev = observation
            reward = 0
            while not terminal:
                # render method will print you the situation in the terminal
                # self.env.render()
                action = self.receive_observation_and_get_action(observation, terminal)
                observation, reward, terminal, _, _ = self.env.step(action)
                # TODO your code will be very likely here
                self.update_q_values(prev, action, reward, observation)
                prev = observation
            self.curr_state[i] = self.state_action_values['13,6,False']

        plt.plot(self.curr_state)
        plt.savefig('currstate.png')
        print("MY Q:"+str(self.curr_state[self.number_of_episodes-1]))
            # self.env.render()

    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        return BlackjackAction.HIT.value if observation.player_hand.value() < 17 else BlackjackAction.STAND.value

    def update_q_values(self, prev: BlackjackObservation, action: int, reward: float,
                        observation: BlackjackObservation):
        """
        Updates the Q-values using the temporal difference learning rule.
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
        if state not in self.state_action_values:
            self.state_action_values[state] = 0.0
        if prevstate not in self.state_action_values:
            self.state_action_values[prevstate] = 0.0
        if state not in self.state_visits:
            self.state_visits[state] = 0
        if prevstate not in self.state_visits:
            self.state_visits[prevstate] = 0
        td_target = reward + self.gamma * self.state_action_values[state]
        td_error = td_target - self.state_action_values[prevstate]
        learning_rate = self.alpha/(self.alpha + self.state_visits[prevstate])
        self.state_visits[prevstate] += 1
        self.state_action_values[prevstate] += learning_rate * td_error

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned U value for
        particular observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :return: The learned U-value for the given observation.
        """
        if "ACE" in str(observation.player_hand):
            has_ace = True
        else:
            has_ace = False
        state = str(observation.player_hand.value()) + "," + str(observation.dealer_hand.value()) + "," + str(has_ace)
        return self.state_action_values[state]

