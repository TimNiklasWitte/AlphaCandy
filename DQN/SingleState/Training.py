import numpy as np

from Agent import *

import sys
# sys.path.append("../..")
sys.path.append("../AlphaCandy")
from CandyCrushGym import *


def main():

    batch_size = 16

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episods = 500000
    update = 250

    env_field_size = 6
    env_num_elements = 6

    # Init gym
    envs = gym.vector.AsyncVectorEnv([
            lambda: CandyCrushGym(1000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(2000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(3000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(4000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(5000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(6000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(7000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(8000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(9000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(10000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(11000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(12000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(13000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(14000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(15000, env_field_size, env_num_elements),
            lambda: CandyCrushGym(16000, env_field_size, env_num_elements),
        ])
    
    
    agent = Agent(input_dims=(*envs.observation_space.shape, 26),
                   num_actions=255, batch_size=batch_size)
    agent.q_net.summary()

    # Start actual training
    with train_summary_writer.as_default():

        for episode in range(num_episods):


            score = 0  # sum of rewards
            rewards_list = []


            states = envs.reset()
            for i in range(50):

                actions = agent.select_action(states)
                next_states, rewards, _, _ = envs.step(actions)


                for batch_idx in range(batch_size):

                    if rewards[batch_idx] == 0:
                        x = np.random.randint(0, 10)
                        if x == 0:
                            agent.store_experience(states[batch_idx], actions[batch_idx], next_states[batch_idx], rewards[batch_idx])

                    else:
                        agent.store_experience(states[batch_idx], actions[batch_idx], next_states[batch_idx], rewards[batch_idx])


                states = next_states
                agent.train_step()

                score += np.mean(rewards)

                rewards_list.append(np.mean(rewards))


            # Reduce epsilon after each episode
            agent.strategy.reduce_epsilon()

            # Update target network
            if episode % update == 0:
                agent.update_target()

            # Save weights
            if episode % 250 == 0:
                agent.q_net.save_weights(f"./saved_models/trained_weights_episode_{episode}", save_format="tf")

            tf.summary.scalar(f"Average reward (DQN_SingleState)", np.mean(rewards_list), step=episode)
            tf.summary.scalar(f"Score (DQN_SingleState)", score, step=episode)
            tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy) (DQN_SingleState)", agent.strategy.get_exploration_rate(), step=episode)


            print(f"   Episode: {episode}")
            print(f"   Epsilon: {round(agent.strategy.get_exploration_rate(), 2)}")
            print(f"     Score: {round(score, 2)}")
            print(f"Avg Reward: {round(np.mean(rewards_list), 2)}")
            print("------------------------")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
