import tensorflow as tf

class DecisionTransformer(tf.keras.Model):
    def __init__(self, episode_len, num_actions):
        super(DecisionTransformer, self).__init__()

        self.episode_len = episode_len


        self.feature_extractor = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='same')
        self.bottleneck = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation="tanh", padding='same')
        self.embedding = tf.keras.layers.Dense(10, activation=tf.nn.tanh)

        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=5, key_dim=20, value_dim=10)
        self.backend_layer_list = [
            tf.keras.layers.Dense(100, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_actions, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
    def call(self, game_states, rewards):

        batch_size = game_states.shape[0]
        field_size = game_states.shape[2]

        # Extract features from each game state
        game_states_embeddings = []

        for episode_idx in range(self.episode_len):

            game_state = game_states[:, episode_idx, :, :]

            # feature_extractor -> bottleneck -> embedding
            embedding = self.feature_extractor(game_state)
            embedding = self.bottleneck(embedding)
            embedding = self.embedding(embedding)

            # embedding: flatten
            embedding = tf.reshape(embedding, shape=(batch_size, 3*3*10))

            # collect embedding for each time step
            game_states_embeddings.append(embedding)
            

        # Merge embedding for each time step
        game_state_embeddings = tf.stack(game_states_embeddings,axis=1)

        #
        # Merge embedding of game state with rewards
        #
        rewards = tf.expand_dims(rewards, axis=-1)
    
        input = tf.concat([game_state_embeddings, rewards], axis=-1)
        
        x = self.self_attention(input, value=input)

        #
        # pass through backend
        #

        for layer in self.backend_layer_list:
            x = layer(x)

          
        return x

    @tf.function
    def train_step(self, game_states, rewards, actions_target):

        with tf.GradientTape() as tape:
            actions_prediction = self(game_states, rewards)
            loss = self.loss_function(actions_target, actions_prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        actions_prediction = tf.argmax(actions_prediction, axis=-1)
        actions_target = tf.argmax(actions_target, axis=-1)
        self.metric_accuracy.update_state(actions_target, actions_prediction)
    
    #@tf.function
    def test_step(self, dataset):
        
        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

        for game_states, actions_target, rewards in dataset:
            actions_prediction = self(game_states, rewards)

            loss = self.loss_function(actions_target, actions_prediction)
            self.metric_loss.update_state(loss)

            actions_prediction = tf.argmax(actions_prediction, axis=-1)
            actions_target = tf.argmax(actions_target, axis=-1)
            self.metric_accuracy.update_state(actions_target, actions_prediction)