from EnvManager import *


def main():

    env = EnvManager()

    capacity = 2500000
    
    num_episodes = 15000
    episode_len = 100

    states = np.zeros((capacity, *env.observation_space_shape), dtype=np.int8)
    actions = np.zeros(capacity, dtype=np.int8)
    next_states = np.zeros((capacity, *env.observation_space_shape), dtype=np.int8)
    rewards = np.zeros(capacity, dtype=np.float32)
    
    i = 0
    for _ in range(num_episodes):
        state = env.reset()
     
        for _ in range(episode_len):
            
            isValid = False 
            reward = 0
            action = -1

            while reward == 0:

                while not isValid:
                    action = np.random.randint(0, env.action_space_n)
                    isValid = env.isValidAction(action)

                next_state, reward = env.gym_step(action)

                if reward == 0:
                    x = np.random.randint(0, 10)
                    if x == 0:
                        break 

                else:
                    break
                
                isValid = False 

            next_state = env.save(next_state)
    
            states[i] = state 
            actions[i] = action
            next_states[i] = next_state
            rewards[i] = reward
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