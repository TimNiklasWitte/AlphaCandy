import numpy as np
import tensorflow as tf
import tqdm
import argparse
from DecisionTransformer import *

episode_len = 10
batch_size = 32

def loadData(train_size, num_candys, episode_len, field_size):

    none_action_id = field_size*field_size*4 + 1

    buff_states = np.load(f"./TrainingData/states_{field_size}_{num_candys}_{train_size}.npy")
    buff_rewards = np.load(f"./TrainingData/rewards_{field_size}_{num_candys}_{train_size}.npy")
    buff_actions = np.load(f"./TrainingData/actions_{field_size}_{num_candys}_{train_size}.npy")

    for buff_idx in range(train_size):

        for episode_idx in range(episode_len):
            states = np.copy(buff_states[buff_idx])
            states[episode_idx + 1:episode_len, :, :] = 0

            actions = np.copy(buff_actions[buff_idx])
            actions[episode_idx:episode_len] = none_action_id

            rewards = np.copy(buff_rewards[buff_idx])
            rewards[episode_idx + 1:episode_len] = 0

            action_target = buff_actions[buff_idx, episode_idx]
          
            yield states, actions, rewards, action_target
     

def checkFieldSize(size: str):
    size = int(size)
    if size <= 4:
        raise argparse.ArgumentTypeError("Field size must be greater than 4")
    return size

def checkNumCandys(num: int):
    num = int(num)
    if num <= 3:
        raise argparse.ArgumentTypeError("Number of candys must be greater than 3")
    return num

def checkTrainSize(size: str):
    size = int(size)
    if size <= 0:
        raise argparse.ArgumentTypeError("Number of training samples must be greater than 0")
    return size

def checkEpochs(size: str):
    size = int(size)
    if size <= 0:
        raise argparse.ArgumentTypeError("Number of epochs must be greater than 0")
    return size

def main():

    parser = argparse.ArgumentParser(description="Create Training data.")
    parser.add_argument("--fieldSize", help="Set the field size.", type=checkFieldSize, required=True)
    parser.add_argument("--numCandys", help="Set the number of candys.", type=checkNumCandys, required=True)
    parser.add_argument("--trainSize", help="Set the number of training samples.", type=checkTrainSize, required=True)
    parser.add_argument("--epochs", help="Set the number of epochs.", type=checkEpochs, required=True)
    args = parser.parse_args()


    field_size = args.fieldSize
    num_candys = args.numCandys
    train_size = args.trainSize
    num_epochs = args.epochs
    

    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    dataset = tf.data.Dataset.from_generator(loadData,
        args=(train_size, num_candys, episode_len, field_size,), 
        output_signature=(
            tf.TensorSpec(shape=(episode_len, field_size, field_size), dtype=tf.uint8), # states
            tf.TensorSpec(shape=(episode_len,), dtype=tf.uint8),  # actions
            tf.TensorSpec(shape=(episode_len,), dtype=tf.float32), # rewards
            tf.TensorSpec(shape=(), dtype=tf.uint8),  # action_target
        )
    )

    #
    # Dataset
    #
    test_size = 10000
    train_dataset = dataset.take(train_size - test_size)
    train_dataset = train_dataset.apply(lambda x: prepare_data(x, field_size))

    test_dataset = dataset.take(test_size)
    test_dataset = test_dataset.apply(lambda x: prepare_data(x, field_size))
    
    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
    
    log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, 0, field_size, num_candys)

    #
    # Train loop
    #

    for epoch in range(num_epochs):
            
        print(f"Epoch {epoch}")

        for states, actions, rewards, action_target in tqdm.tqdm(train_dataset, total=int(train_size/batch_size)): 
            decisionTransformer.train_step(states, actions, rewards, action_target)

        log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch + 1, field_size, num_candys)
    
    decisionTransformer.save_weights(f"./saved_models/trained_weights_{field_size}_{num_candys}", save_format="tf")

def prepare_data(data, field_size):
    
    global episode_len
    global batch_size 

    #
    # onehotify states
    #

    # state shape: (episode_len, field_size, field_size) -> (episode_len * field_size * field_size)
    data = data.map(lambda states, actions, rewards, action_target: (tf.reshape(states, shape=(episode_len*field_size*field_size,)), actions, rewards, action_target))

    # one hot
    num_one_hot = 26 # num of candys
    data = data.map(lambda states, actions, rewards, action_target: (tf.one_hot(states, depth=num_one_hot), actions, rewards, action_target))

    # state shape: (episode_len * field_size * field_size) -> (episode_len, field_size, field_size)
    data = data.map(lambda states, actions, rewards, action_target: (tf.reshape(states, shape=(episode_len, field_size, field_size, num_one_hot)), actions, rewards, action_target))


    #
    # onehotify actions
    #

    num_actions = field_size*field_size*4 + 1
    data = data.map(lambda states, actions, rewards, action_target: (states, tf.one_hot(actions, depth=num_actions), rewards, action_target))

    #
    # onehotify action_target
    #

    data = data.map(lambda states, actions, rewards, action_target: (states, actions, rewards, tf.one_hot(action_target, depth=num_actions)))

    #
    # cache, shuffle, batch, prefetch
    # 
    #data = data.cache()
 
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch, field_size, num_candys):

    with train_summary_writer.as_default():
        
        if epoch == 0:
            decisionTransformer.test_step(train_dataset.take(1000))

   
        loss = decisionTransformer.metric_loss.result()
        tf.summary.scalar(f"train_loss_{field_size}_{num_candys}", loss, step=epoch)
        
        accuracy = decisionTransformer.metric_accuracy.result()
        tf.summary.scalar(f"train_accuracy_{field_size}_{num_candys}", accuracy, step=epoch)

        print(f"train_loss: {loss}")
        print(f"train_accuracy: {accuracy}")

        decisionTransformer.metric_loss.reset_states()
        decisionTransformer.metric_accuracy.reset_states()


        decisionTransformer.test_step(test_dataset)

        loss = decisionTransformer.metric_loss.result()
        tf.summary.scalar(f"test_loss_{field_size}_{num_candys}", loss, step=epoch)

        accuracy = decisionTransformer.metric_accuracy.result()
        tf.summary.scalar(f"test_accuracy_{field_size}_{num_candys}", accuracy, step=epoch)

        print(f"test_loss: {loss}")
        print(f"test_accuracy: {accuracy}")

        decisionTransformer.metric_loss.reset_states()
        decisionTransformer.metric_accuracy.reset_states()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")