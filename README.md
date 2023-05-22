# CryptoEnvironment

CryptoEnvironment is a gym environment for cryptocurrency trading. It provides a simulation environment for training and evaluating trading strategies.

## Features

- Supports buying, selling, and holding actions for trading.
- Customizable initial balance, random initial asset split, maximum steps, trade fees, slippage, order fraction, and reward function.
- Uses any trading data including custom indicators from a provided DataFrame.
- Provides observation and action spaces compatible with Gym.
- Easy integration with stable-baselines3
- Supports one-line evaluation of trained models with visualization options.


## Installation

To use the CryptoEnvironment, you need to have the following dependencies installed:

- gym==0.21.0
- stable-baselines3==1.8.0
- matplotlib==3.7.1
- numpy
- pandas

You can install these dependencies using pip:
```bash
pip install -r requirements.txt
```

Clone the repository:
```bash
git clone https://github.com/your-username/crypto-trading-gym.git
```

## Usage

Here's an example of how to use the CryptoEnvironment:

```python
# Imports
import requests
import pandas as pd
import matplotlib.pyplot as plt 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from tradinggym import CryptoEnvironment

# Load the data
data = pd.DataFrame(requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30")
data = data.json()["prices"], columns=["Timestamp", "Close"])

# Create the CryptoEnvironment
environment = CryptoEnvironment(observations=data, initial_value = 10000, 
                                window_size=5, order_fraction=0.66, trade_fee=0.003375)

# Wrap the training environment in a vectorized form
environment = DummyVecEnv([lambda: environment])

# Create a model
model = PPO('MlpPolicy',env=environment)

# Train the model
model.learn(10000)

# Create an evaluation environment
testenv = CryptoEnvironment(observations=data, initial_value = 10000, 
                            window_size=5, order_fraction=0.66, trade_fee=0.003375)

# Evaluate the model
testenv.evaluate(frame_length=50, render=True, model=model, marker_size=25, verbose=1)
```
![07c519cc-f91c-41bb-af8b-0b0887f5295e](https://github.com/astrologos/tradinggym/assets/82430396/83342eda-5b86-463c-b5b1-861b72d90268)

Evaluation Metrics:  
- Initial balance:     10000.00
- Final value:         9929.66
- Profit:              -70.34
- Return Rate:         -0.70%
- Cumulative reward:   0.54
- Max reward:          1.89
- Min reward:          -2.95

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
