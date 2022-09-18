# Play Candy Crush by applying Deep Reinforcement Learning
# "Large Action Spaces are Problematic in Deep Reinforcement Learning"
## by Marc Zeller, Robin Gratz and Tim Niklas Witte 


This repository contains three approaches to playing the game of Candy Crush.
Said approaches consist of a Deep Q-Network (DQN), a Proximal Policy Optimization (PPO) algorithm
and a Decision Transformer.
This repository also contains a parameterizable Candy Crush environment wrt. to the total field size and the number of candies.
It turns out that both DQN and PPO are unable to learn the Candy Crush game succefully.
Only the Decision Transformer is able to produce good results.

<img src="./media/play_game_mode_0.gif" width="300" height="300">

Video: [Project video summary](https://www.youtube.com/watch?v=OzU-eDqEk-g)

## Requirements
- TensorFlow 2
- TensorboardX
- Numpy
- gym
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

This initializes a Replay Memory with 500.000 samples.
It creates the following files `states_500000`, `actions_500000`, `next_states_500000` and 
`rewards_500000`.
These files will be loaded by the `ReplayMemory.py` script when `Training.py` is launched.

After `ReplayMemory.py` is launched, `Training.py` must be started.

```
python3 Training.py
```

The DQN will be trained. 
Note that the ReplayMemory has a capacity (maximum amount of training samples) of 1,000,000.

#### PPO

Run `Train.py` with the desired configurations.

```
usage: Train.py [-h] --size SIZE --num NUM

Setup the desired CandyCrush/Training sconfiguration

optional arguments:
  -h, --help   show this help message and exit
  --size SIZE  Set the field size.
  --num NUM    Set the number of candys.
```

#### Decision Transformer

First run `GenerateTrainingData.py`.
This script will generate the training data as configured (program arguments):

```
usage: GenerateTrainingData.py [-h] --fieldSize FIELDSIZE --numCandys
                               NUMCANDYS --trainSize TRAINSIZE

Create Training data.

optional arguments:
  -h, --help            show this help message and exit
  --fieldSize FIELDSIZE
                        Set the field size.
  --numCandys NUMCANDYS
                        Set the number of candys.
  --trainSize TRAINSIZE
                        Set the number of training samples.
```

Now run `Training.py`. It will train the Decision Transformer.
It must be launched with the same configuration as `GenerateTrainingData.py`.

```
usage: Training.py [-h] --fieldSize FIELDSIZE --numCandys NUMCANDYS
                   --trainSize TRAINSIZE --epochs EPOCHS

Create Training data.

optional arguments:
  -h, --help            show this help message and exit
  --fieldSize FIELDSIZE
                        Set the field size.
  --numCandys NUMCANDYS
                        Set the number of candys.
  --trainSize TRAINSIZE
                        Set the number of training samples.
  --epochs EPOCHS       Set the number of epochs.
```


### Window 

Run `PlayGame.py` with desired configuration to watch the Decision Transformer play the game.
The corresponding pretrained weights will be loaded.
Note that, for each `field_size` and `num_candys` there are different pretrained weights.
Besides that, a GIF can also be created.

```
usage: PlayGame.py [-h] [--field_size FIELD_SIZE] [--num_candys NUM_CANDYS]
                   [--desired_reward DESIRED_REWARD] [--mode MODE] [--gif GIF]

The Decision Transformer plays Candy Crush.

optional arguments:
  -h, --help            Show this help message and exit
  --field_size FIELD_SIZE
                        Set the field size (default = 8).
  --num_candys NUM_CANDYS
                        Set the number of candys (default = 4).
  --desired_reward DESIRED_REWARD
                        Set the desired reward (default = 0.25)
  --mode MODE           Define the window mode (default: "0") "0" = game
                        window or "1" = game window with plots
  --gif GIF             File path where the GIF (screenshots of the window)
                        will be saved.
```

#### mode = 0

<img src="./media/play_game_mode_0.gif" width="300" height="300">

#### mode = 1

<img src="./media/play_game_mode_1.gif" width="1700" height="400">

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
