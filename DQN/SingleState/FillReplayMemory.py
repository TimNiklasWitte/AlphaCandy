import sys
sys.path.append("../..")
from CandyCrushGym import *

import tqdm


import gym

def sample_actions(num_actions):

    actions = []
    for i in range(num_actions):

        while True:
            action = np.random.randint(0, 255)

            fieldID = action // 4

            direction = action % 4

            x = fieldID // 8
            y = fieldID % 8

            # Swap candy
            x_swap = x # attention: numpy x->y are swapped
            y_swap = y # attention: numpy x->y are swapped
            # top
            if direction == 0:
                y_swap += -1
            # down
            elif direction == 2: 
                y_swap += 1
            # right 
            elif direction == 1:
                x_swap += 1
            # left 
            elif direction == 3:
                x_swap += -1

            if isValidIndex(x,y) and isValidIndex(x_swap, y_swap):
                break
        
    
        actions.append(action)

    return np.array(actions)

def isValidIndex(x, y):
        if 8 <= x or 8 <= y or x < 0 or y < 0:
            return False
        return True

def main():
    
    envs = gym.vector.AsyncVectorEnv([
            lambda: CandyCrushGym(100),
            lambda: CandyCrushGym(200),
            lambda: CandyCrushGym(300),
            lambda: CandyCrushGym(400)
        ])

    actions = sample_actions(4)

    print(actions)
    envs.step(actions)
    
   
    return 
    
    # print("here")
    # state1, state2, state3 = envs.reset()

    # print(state1)
    # print(state2)
    # print(state3)
    #print(envs.observation_space)

    return 
    env = CandyCrushGym()

    capacity = 2500000
    
    num_episodes = 1000
    episode_len = 100

    states = np.zeros((capacity, *env.observation_space_shape), dtype=np.int8)
    actions = np.zeros(capacity, dtype=np.int8)
    next_states = np.zeros((capacity, *env.observation_space_shape), dtype=np.int8)
    rewards = np.zeros(capacity, dtype=np.float32)
    
    i = 0
    for _ in tqdm.tqdm(range(num_episodes), position=0, leave=True):
        state = env.reset()
     
        for _ in range(episode_len):
            
            isValid = False 
            reward = 0
            action = -1

            while reward == 0:

                while not isValid:
                    action = np.random.randint(0, env.action_space_n)
                    isValid = env.isValidAction(action)

                next_state, reward = env.step(action)

                if reward == 0:
                    x = np.random.randint(0, 10)
                    if x == 0:
                        break 

                else:
                    break
                
                isValid = False 

            states[i] = state 
            actions[i] = action
            next_states[i] = next_state
            rewards[i] = reward

            state = next_state
            i += 1
            
    np.save(f"./ReplayMemoryData/states_{i}", states)
    np.save(f"./ReplayMemoryData/actions_{i}", actions)
    np.save(f"./ReplayMemoryData/next_states_{i}", next_states)
    np.save(f"./ReplayMemoryData/rewards_{i}", rewards)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")