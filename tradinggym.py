import gym
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gym import spaces

class CryptoEnvironment(gym.Env):
    def __init__(self, initial_value=10000, observations=None, max_steps=100, random_split=True, window_size=5, trade_fee=0.0045, slippage = 0.005, order_fraction = 0.2, reward_function = None, price_column = 'Close'):
        """
        Gym environment for cryptocurrency trading.

        Parameters:
        - initial_value (float): Initial balance for the agent.
        - observations (DataFrame): DataFrame containing observation data.
        - max_steps (int): Maximum number of steps to take in the environment.
        - random_split (float): Asset split is random after each reset() 
        - window_size (int): Window size to feed the model.
        - trade_fee (float): Fees for each trade on (0,1).
        - slippage (float): Maximum slippage coefficient.
        - order_fraction (float): Ratio of maximum possible order to execute on (0,1).
        - reward_function (function): Custom reward function.
        - price_column (str): Name of the column containing price data.
        """
        self.DEBUG = False

        if observations is None:
            import requests
            observations = pd.DataFrame(requests.get('https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1000').json()['prices'], columns=['TimeStamp','Close'])
        
        super(CryptoEnvironment, self).__init__()
        self.initial_value = initial_value  # Initial balance
        self.observations = observations  # Observation dataframe, should contain price_column
        self.window_size = window_size  # Window size to feed model
        self.balance = initial_value  # Balance = initial_value
        self.shares = 0.0  # Fractional shares
        self.random_split = random_split # Random split 
        self.max_steps = max_steps  # Max number of steps to take
        self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1))  # Initialize current step randomly
        self.trade_fee = trade_fee  # Fees for each trade
        self.slippage = slippage  # Maximum slippage coefficient, real coefficient is determined randomly on the interval +/-slippage)
        self.order_fraction = order_fraction  # Ratio of maximum possible order to execute  
        self.price_column = price_column # Name of the column containing price data
        self.rewards = list([i for i in range(window_size)])

        if self.observations is None:
            raise ValueError("Observation data is missing.")
        
        if len(self.observations) <= 2*self.window_size:
            raise ValueError("Observation data must longer than double than the window size.")


        # Define the action and observation spaces
        num_columns = self.observations.shape[1] - 1  # Exclude the price column
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size, num_columns), dtype=np.float32)

        # User may supply a custom reward function
        if reward_function is not None: 
            self._get_reward = reward_function 


    def flush_print(self, s):
        """
        Simple utility to flush print statements.
        """
        if self.DEBUG:
            print(s)
            sys.stdout.flush()
        else: pass

    print = flush_print

    # Reset the state of the environment
    def reset(self, start_index = None):
        """
        Reset the state of the environment

        Returns:
        - observation (ndarray): The initial observation.
        """
        if self.random_split:
            self.balance = self.random_split * self.initial_value
            current_price = self.observations.iloc[self.current_step][self.price_column]
            self.shares = (self.initial_value - self.balance)/ (current_price)
        else:
            self.balance = self.initial_value  # Reset balance
            self.shares = 0.0  # Reset shares
        if start_index is None:
            self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1))  # Initialize current step randomly
        elif start_index > 1 + 2*self.window_size or start_index < len(self.observations) - self.window_size - self.max_steps:
            raise ValueError('Initial step must be between (1 + 2*window_size, len(observations) - window_size - max_steps). \n' + 
            'It is not advised to evaluate the model near the bounds of the observation data.')
        else: self.current_step = start_index
        return self._get_observation()  # Return observation

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        - action (int): The action to take.

        Returns:
        - observation (ndarray): The new observation after the step.
        - reward (float): The reward for the action.
        - done (bool): Indicates if the episode is done.
        - info (dict): Additional information.
        """
        current_price = self.observations.iloc[self.current_step][self.price_column]
        total_fees = self.trade_fee + np.random.uniform(-self.slippage, self.slippage)
        if action == 0:  # Buy
                if self.balance > 0:
                    # Calculate the number of shares we can afford to buy
                    afford_shares = self.order_fraction * (self.balance / (current_price * (1 + total_fees)))
                    # Execute the trade
                    self.shares += afford_shares
                    self.balance -= afford_shares * (current_price * (1 + total_fees))

        elif action == 1:  # Sell
                if self.shares > 0:
                    # Calculate the number of shares we can afford to sell
                    sell_shares = self.order_fraction * self.shares  # Limit the maximum number of shares to sell based on the order fraction
                    self.balance += sell_shares * (current_price * (1 - total_fees))
                    self.shares -= sell_shares

        self.current_step += 1
        done = self.current_step > self.max_steps
        return self._get_observation(), self._get_reward(action), done, {}


    def _get_observation(self):
        """
        Get the current observation.

        Returns:
        - observation (ndarray): The current observation.
        """
        start_index = self.current_step - self.window_size
        end_index = self.current_step
        window_data = self.observations.iloc[start_index-self.window_size:end_index].drop(self.price_column, axis = 1)  # Exclude the price column
    
        # Min-Max scale the windowed data
        scaled_df = pd.DataFrame({col: (window_data[col] - window_data[col].min()) / (window_data[col].max() - window_data[col].min() + 1e-8) for col in window_data.columns})
        return scaled_df.values[self.window_size:]

            ### TODO: add balance and shares to observations


    def _get_reward(self,action):
        """
        Get the reward for the current step.

        Returns:
        - reward (float): The reward for the current step.
        """
        total_fees = self.trade_fee + np.random.uniform(-self.slippage, self.slippage)
        current_price = self.observations.iloc[self.current_step][self.price_column]
        prev_price = self.observations.iloc[self.current_step - 1][self.price_column]
        prev_portfolio_value = self.balance + (((1 - total_fees)* prev_price) * self.shares)
        current_portfolio_value = self.balance + (((1 - total_fees)* current_price) * self.shares)

        n1_price = self.observations.iloc[self.current_step+1][self.price_column]
        n2_price = self.observations.iloc[self.current_step+2][self.price_column]

        n1_portfolio_value = self.balance + (n1_price * self.shares)
        n2_portfolio_value = self.balance + (n2_price * self.shares)

        percent_diff = (1000*(current_portfolio_value - prev_portfolio_value) / prev_portfolio_value)

        return percent_diff 

    def evaluate(self, frame_length, start_index = None, render=True, model=None, deterministic=True, marker_size=20, initial_value=10000, initial_shares=0, verbose = 1, figsize=(14,6)):
        """
        Evaluate the trained model in the environment.

        Parameters:
        - frame_length (int): Number of steps to evaluate.
        - render (bool): Indicates if the evaluation should be visualized.
        - model (object): The trained model to use for action selection.
        - deterministic (bool): Indicates if the model's actions should be deterministic.
        - marker_size (int): Size of the markers in the visualization.
        - initial_value (float): Initial balance for the evaluation.
        - initial_shares (float): Initial shares for the evaluation.
        - verbose (int): Level of verbosity for evaluation metrics.
        - figsize (tuple): Figure size for the visualization.
        """
        initial_value = self.balance 
        initial_shares = self.shares
        initial_step = self.current_step 

        # Reset the environment for evaluation
        self.reset()

        # Randomly select a subset of the data for evaluation
        if start_index is None:
            self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1) - frame_length)  # Initialize current step randomly
        elif start_index < 1 + 2*self.window_size or start_index > len(self.observations) - self.max_steps - 2*(self.window_size + 1) - frame_length:
            raise ValueError('Initial step must be on the interval (1 + 2*window_size, len(observations) - max_steps - 2*window_size - frame_length. \n)' + 
            'It is not advised to evaluate the model near the bounds of the observation data.')
        else: self.current_step = start_index
        eval_start_index = self.current_step + self.window_size
        eval_end_index = eval_start_index + frame_length
        eval_data = self.observations.iloc[eval_start_index:eval_end_index].reset_index(drop=True)

        fig = None

        # Price subplot
        if render:
            fig, axs = plt.subplots(3, figsize=figsize, sharex=True, height_ratios=[3,1,1])
            axs[0].plot(range(0,len(eval_data[self.price_column])), eval_data[self.price_column], label=('Close Price'))
            plt.suptitle('Trading Evaluation')
            plt.xlabel('Time Step')
            axs[0].set_ylabel('Price')
            axs[0].grid(visible=True, alpha = 0.5)

        profit = 0.0  # Accumulated profit
        observation = self._get_observation()  # Get the initial observation
        current_price = None
        rewards = []
        portfolio_vals = []

        for index, row in eval_data.iterrows():
            # Perform action using the trained model
            action, _ = model.predict(observation, deterministic=deterministic)
            action = int(action)

            # Step through the environment with the selected action
            next_observation, reward, done, _ = self.step(action)

            # Accumulate the rewards
            current_price = row[self.price_column]
            rewards.append(self._get_reward(action))
            portfolio_vals.append(self.balance + ((1 - self.trade_fee)*current_price*self.shares))

            # Price subplot
            if render:
                # Plot marker based on action
                if action == 0:  # Buy
                    axs[0].scatter(index, current_price, color='green', marker='o', s=marker_size)
                elif action == 1:  # Sell
                    axs[0].scatter(index, current_price, color='red', marker='o', s=marker_size)
                else: axs[0].scatter(index, current_price, color='blue', marker='o', s=marker_size)

            # Update the observation for the next step
            observation = next_observation


        final_balance = self.balance 
        profit = (final_balance + (current_price*self.shares)) - self.initial_value
        return_rate = ((self.initial_value + profit) / self.initial_value) - 1
        final_value = initial_value + profit

        # Portfolio value subplot
        if render: 
            axs[1].set_ylabel('Portfolio Value')
            axs[1].plot(portfolio_vals, color='#15ab5b', lw=2)
            axs[1].grid(visible=True, alpha = 0.5)

        # Reward subplot
        if render:
            axs[2].set_ylabel('Reward')
            axs[2].plot(rewards, color='#e89f0c',lw=2)
            axs[2].grid(visible=True, alpha = 0.5)

        if render:
            plt.show()

        # Restore the environment state to its original values
        self.balance = initial_value
        self.shares = initial_shares
        self.current_step = initial_step

        # Print evaluation metrics
        if verbose>0:
            print("Evaluation Metrics:  ")
            print("Initial balance:     " + format(initial_value,'.2f'))
            print("Final value:         " + format(final_value, '.2f'))
            print("Profit:              " + format(profit,'.2f'))
            print("Return Rate:         " + format(return_rate,'.2%'))
            print("Cumulative reward:   " + format(sum(rewards),'.2f'))
            print("Max reward:          " + format(max(rewards),'.2f'))
            print("Min reward:          " + format(min(rewards),'.2f'))
