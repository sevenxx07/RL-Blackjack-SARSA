# Problem description
Implementing an agent capable of playing a simplified version
of the blackjack game (sometimes called 21-game). However, in this project, we will restrict ourselves only to a simplified version.
The game is played with a standard deck of 52 cards, which is shuffled. Your goal is to score
more than the dealer; however, you do not want to get over 21. In the beginning, you are given
two cards and see one card that the dealer has. You can decide whether you draw one more
card or stop playing. Once you stop playing, it is the dealer’s turn. The dealer has to follow a
fixed strategy — as long as the sum of his cards is less than 17, he has to draw a card. Dealer
stops when this condition becomes false.
Face cards (Jack, Queen, and King) have a value of 10. Ace can be counted as 1 or 11.
At the end of the game, the player loses if the value of his cards exceeds 21. We call this
situation bust. The player loses even if the dealer busts too. If the dealer busts and the player
not, the player wins. If neither the player nor the dealer busts, the winner is determined by the
value of the cards. The player with a higher sum of cards wins. Equal sums mean tie.

# Implementation
I used the Gymnasium [2] library (formerly Open AI Gym) as an environment for the
game. You can find the environment implementation in file blackjack.py. File carddeck.py
contains a model of card, card deck, and player hand. After each step, your agent will get an
observation as an instance of BlackjackObservation class and a reward. In a terminal state,
you get a reward of 1 for winning, −1 for losing, and 0 for a tie. In any other state, you get
zero as a reward. You are not allowed to modify files blackjack.py and carddeck.py. The
same holds for file main.py above the comment stating that you cannot modify the code.

# State representations
 * Represent the state using the sum of the player’s hand value as P and the
 sum of the dealer’s hand value D.- (P, D).  The player’s hand value can range from 4 to 21(since the player can’t bust
 initially), and the dealer’s visible card can range from 2 to 11. So, the
 total number of states is approximately 18×10=180.
 * Represent the state using tuple of cards on the player’s hand (P1, P2) and
 value of dealer’s card D. This representation gives us more information
 about the remaining cards in the deck.- (P1, P2, D) where P1,P2,D ∈
 {1,2,3,4,5,6,7,8,9,10,J,Q,K}. Since there is 13 types of cards and we have 3 variables, the total number
 of states is approximately 13x13x13=2197
 * Represent the state using the sum of the player’s hand value as P, the
 sum of the dealer’s hand value D and usable ace A. The usable ace A is
 boolean telling if the player has usable ace or not which can significantly
 affect the strategy.- (P, D, A).  The player’s hand value can range from 4 to 21, and the dealer’s visible
 card can range from 2 to 11. The usable ace A is boolean. So, the total
 number of states is approximately 18×10x2=360.

Third representation was chosen, because the number of state is not too large
 and the representation captures essential information for decision-making, in
cluding the player’s current hand value, the dealer’s visible card, and whether
 the player has a usable ace.
The chosen state representation captures essential information for agent decisions in the simplified version of the blackjack game. However, there is a
 simplification involved in this representation.
 Omitting specific card details reduces the amount of information available
 to the agent. The agent does not consider the specific cards in the player’s
 or dealer’s hands, which could potentially provide additional insights into the game state.
 The simplification may affect the agent’s learned policy and utility values.
 Without considering individual cards, the agent’s decisions may rely more heavily on general rules or heuristics rather than nuanced card combinations.
 The agent’s policy may become more deterministic or less adaptive due to the reduced information available in the state representation.
 Depending on the complexity of the game dynamics and the chosen simplif ication, the agent’s learning convergence may be slower or less optimal.
 
 # Discount factor and Q values
 The discount factor was chosen to be 0.95 since it is common to use values close
 to one because the game has a fixed horizon (ends after each round).
 The number of games needed to learn the Q-values depends on various fac
tors, including the agent’s exploration strategy, the learning rate, and the com
plexity of the state space. The agents were trained for 50 000- 100 000 epochs.
 Also the decreasing learning rate was chosen to provide better results:
 α = c / c−number of visits
 Parameter c was set equal to 100, as that produced the best results.
 In SARSA algorithm was used exponential decay for greedy value
 ϵi = ϵ * e^(−decay∗epochi)
 Parameter ϵ was set equal to 0.9 as the start of greedy value and decay was set
 equal to 0.001 to slowly decrease the exploration rate.

# Results
 The following table summarizes results of the agents obtained after training
 100K epochs.
 | Agent name | Average result    | 
| :---:   | :---: | 
| Random | -0.393   | 
 | Dealer | -0.083  | 
 | TDAgent | -0.079   | 
 | SARSA | -0.085   | 
  For the 100K epochs shown in previous table, evaluation take around two min
utes for both agents.

 I expected that the average result of Dealer, TD agent and SARSA will be
 nearly the same. But I think SARSA should have slightly better results, but
 they haven’t been achieved even after implementation decreasing learning rate
 and exponential decay of exploration rate.
 
 # utility values estimate/Q∗ values converge
  Since there was used the epsilon decay and decreasing learning rate, which both
 decrease over time and approach the value of 0 in the limit, my utility value
 estimations and Q values converge in an unlimited number of epochs. The
 learning rate falls as the number of visits in state rises and epsilon decay falls
 as the number of epochs increases.
 When the ϵ approaches zero in the limit, the learned Q-function is subject
 to a greedy policy. Verified through computation, we apply the epsilon setting
 and establish the limit
 limi→∞(ϵ * e^(−decay∗epochi)) = 0,
 This makes the agent greedy according to the learned Q values, satisfying GLIE
 conditions.
 Similarly, the learning rate satisfies Robbins-Munro conditions:<br />
 ![obrazek](https://github.com/user-attachments/assets/e7128605-0d78-4c19-a6c7-33668190cd2f)<br />
 ![obrazek](https://github.com/user-attachments/assets/a3a16d53-f719-4dd7-a32c-3e0b70321a5a)<br />
  Even after many learning runs, the algorithm gives similar results of the utility
 value estimations and Q values. However, the values keep oscillating slightly, but
 as the epochs rise the algorithm should converge according to satisfied condition
 described in previous question.

