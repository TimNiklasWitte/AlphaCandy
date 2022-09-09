import tensorflow as tf

class DecisionTransformer(tf.keras.Model):
    def __init__(self, episode_len, num_actions):
        super(DecisionTransformer, self).__init__()

        self.episode_len = episode_len


        self.feature_extractor = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="tanh", padding='same')
        
        self.bottleneck = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation="tanh", padding='same')

        self.layer_list = [
            tf.keras.layers.Flatten,
            tf.keras.layers.Dense(num_actions, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_mean = tf.keras.metrics.Mean(name="loss")


    #@tf.function
    def call(self, x):

        # Extract features from each game state
        feature_maps = []

        for episode_idx in range(self.episode_len):

            game_state = x[:, episode_idx, :, :]

            y = self.feature_extractor(game_state)
            feature_maps.append(y)

        x = tf.concat(feature_maps, axis=-1)

        # Compress features maps
        x = self.bottleneck(x)
        
        print(x)
        exit()
        # for layer in self.layer_list: 
        #     x = layer(x)
  
        return x

    @tf.function
    def train_step(self, input, target):

        with tf.GradientTape() as tape:
            prediction = self(input, training=True)
            loss = self.loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_mean.update_state(loss)
    

    def test(self, test_data):

        self.metric_mean.reset_states()

        # test over complete test data
        for input, target in test_data:           
            prediction = self(input)
            
            loss = self.loss_function(target, prediction)
            self.metric_mean.update_state(loss)

        mean_loss = self.metric_mean.result()
        self.metric_mean.reset_states()
        return mean_loss