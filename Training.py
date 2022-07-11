import numpy as np

from Agent import *
from EnvManager import *


def main():


    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episods = 500000
    update = 250

    # Init gym
    env = EnvManager()
    
    agent = Agent(input_dims=(*env.observation_space_shape, 26),
                   num_actions=env.action_space_n, batch_size=64)
    agent.q_net.summary()


    # Start actual training
    with train_summary_writer.as_default():

        for episode in range(num_episods):


            score = 0  # sum of rewards
            rewards = []

        
            state = env.reset()#, depth=25, axis=-1)
            for i in range(50):
                
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

                #next_state, reward = env.step(action)
                #next_state = tf.one_hot(next_state, depth=25, axis=-1)
                agent.store_experience(state, action, next_state, reward)
             
                state = next_state
                agent.train_step()

                score += reward

                rewards.append(reward)
           

            # Reduce epsilon after each episode
            agent.strategy.reduce_epsilon()

            # Update target network
            if episode % update == 0:
                agent.update_target()

            # Save weights
            # if episode % 10 == 0:
            #     agent.q_net.save_weights(f"./saved_models/trainied_weights_epoch_{episode}", save_format="tf")

            tf.summary.scalar(f"Average reward", np.mean(rewards), step=episode)
            tf.summary.scalar(f"Score", score, step=episode)
            tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy)", agent.strategy.get_exploration_rate(), step=episode)
    

            print(f"  Episode: {episode}")
            print(f"  Epsilon: {round(agent.strategy.get_exploration_rate(), 2)}")
            print(f"    Score: {round(score, 2)}")
            print(f"Avg Score: {round(np.mean(rewards), 2)}")
            print("------------------------")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
