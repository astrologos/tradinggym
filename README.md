[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/ajack)


# CryptoEnvironment

CryptoEnvironment is a gym environment for cryptocurrency trading. It provides a simulation environment for training and evaluating reinforcement learning agents.

## Features

- Supports discrete buying, selling, and holding (optional) actions.
- Customizable:
    - Initial balance
    - Random initial asset split
    - Maximum steps
    - Trade fees
    - Slippage
    - Order execution fraction
    - Reward function
    - Low entropy action penalty
- Uses any trading data including custom indicators from a provided DataFrame.
- Provides observation and action spaces compatible with Gym.
- Easy integration with stable-baselines3.
- Supports one-line evaluation of trained models with visualization and reporting options.


## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/crypto-trading-gym.git
```

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

## Usage
See more detailed examples in `/examples/`.

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
data = pd.read_pickle('./data/measurement.pickle')

# Create the CryptoEnvironment
environment = CryptoEnvironment(observations=data, initial_value = 10000, 
                                window_size=5, order_fraction=0.66, trade_fee=0.003375)

# Wrap the training environment in a vectorized form
environment = DummyVecEnv([lambda: environment])

# Create a model
model = PPO('MlpPolicy',env=environment)

# Train the model
model.learn(1000)

# Create an evaluation environment
testenv = CryptoEnvironment(observations=data, initial_value = 10000, 
                            window_size=5, order_fraction=0.66, trade_fee=0.003375)

# Evaluate the model
testenv.evaluate(frame_length=50, render=True, model=model, marker_size=25, verbose=1)
```

![4acc439c-7742-4282-a167-9c19d234ab7c](https://github.com/astrologos/tradinggym/assets/82430396/d4917963-d8cb-4c24-b595-301d8bf876ff)

```
Evaluation Metrics:  
Initial value:       9999.00
Initial balance:     8728.51
Initial shares:      0.04
Initial split:       0.87
Final value:         10109.60
Profit:              110.60
Return Rate:         1.11%
Avg reward:          -0.00
Max reward:          0.00
Min reward:          -0.00
Cumulative reward:   -0.13
```

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
