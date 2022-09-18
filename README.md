# Play Candy Crush by applying Deep Reinforcement Learning
# "Large Action Spaces are Problematic in Deep Reinforcement Learning"
## by Marc Zeller, Robin Gratz and Tim Niklas Witte 


This repository contains three approaches to play the game Candy Crush.
These three approaches are a Deep Q-Network (DQN), a Proximal Policy Optimization (PPO) algorithm
and a Decision Transformer.
This repository also contains a parameterizable Candy Crush field size and number of candies.
It turn out that DQN and PPO are unable to learn the Candy Crush game.
Only the Decision Transformer is able to learn it.

<img src="./media/play_game_mode_0.gif" width="300" height="300">


## Requirements
- TensorFlow 2
- Numpy
- tkinter
- matplotlib
- argparse
- imageio
- pyautogui

## Usage

### Training

#### DQN

First run `ReplayMemory.py` 

```
python3 ReplayMemory.py
```

It initialize a Replay Memory with 500.000 samples.
It creates the following files `states_500000`, `actions_500000`, `next_states_500000` and 
`rewards_500000`.
These files will be loaded by the `ReplayMemory.py` when `Training.py` is launched.

After `ReplayMemory.py` was launched, `Training.py` must be started.

```
python3 Training.py
```

### Window 

## Results

### DQN
#### Average Reward
<img src="./DQN/SingleState/Plots/average_reward.png" width="500" height="400">

#### Score
<img src="./DQN/SingleState/Plots/score.png" width="500" height="400">

### PPO
#### Average Reward
<img src="./PPO/pictures/PPO_AvgReward.png" width="500" height="400">

#### Score
<img src="./PPO/pictures/PPO_Score.png" width="500" height="400">

### Decision Transformer
#### Training (Accuracy on test dataset)
<img src="./Decision Transformer/Plots/DesiredRewardPerformance.png" width="500" height="400">

#### Error of desired reward and achieved reward (absolute value)
<img src="./Decision Transformer/Plots/ErrorDesiredRewardPerformance.png" width="500" height="400">