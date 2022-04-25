# https://github.com/ejmejm/CartPole-RL-DNN/blob/master/DeepNNRollout.ipynb
# https://stackoverflow.com/questions/56904270/difference-between-openai-gym-environments-cartpole-v0-and-cartpole-v1
import gym
import numpy as np
import matplotlib
import tflearn
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class NeuralNetwork:
    def __init__(self, inputs:int, outputs:int):
        self.observation = tflearn.input_data(shape=[None, inputs])  # input layer; should be the entire observation
        net = tflearn.fully_connected(self.observation, 256, activation="relu")  # hidden layers
        net = tflearn.fully_connected(net, 256, activation="relu")  # hidden layers
        net = tflearn.fully_connected(net, 256, activation="relu")  # hidden layers
        self.out = tflearn.fully_connected(net, outputs, activation="softmax")

        self.reward_holder = tf.placeholder(tf.float32, [None])
        self.action_holder = tf.placeholder(tf.int32, [None])

        self.responsible_outputs = tf.gather(tf.reshape(self.out, [-1]), tf.range(0, tf.shape(self.out)[0] * tf.shape(self.out)[1], 2) + self.action_holder)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        self.optimizer = tf.train.AdamOptimizer()
        self.update = self.optimizer.minimize(self.loss)

class Agent:
    def __init__(self):
        self.discount_factor = 0.99

        self.num_episodes = 1500
        self.max_time = 500
        self.all_rewards = []
        self.saver = tf.train.Saver()
        self.train_data = []

        self.nn = NeuralNetwork(4, 2)
        
    def discount_reward(self, rewards):
        running_total = 0.0
        result = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            result[i] = rewards[i] + self.discount_factor * running_total
            running_total += rewards[i]
        return result

    def run_agent(self, env):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_episodes):
                obs = env.reset()
                episode_reward = 0
                ep_history = []
                for j in range(self.max_time):
                    a_one_hot = sess.run(self.nn.out, feed_dict={self.nn.observation: [obs]}).reshape(2)
                    action = np.random.choice(a_one_hot, p=a_one_hot)
                    action = np.argmax(a_one_hot == action)
                    obs1, r, d, _ = env.step(action)
                    ep_history.append([obs, r, action])
                    obs = obs1
                    episode_reward += r
                    if d == True:
                        self.all_rewards.append(episode_reward)
                        ep_history = np.array(ep_history)
                        ep_history[:, 1] = self.discount_reward(ep_history[:, 1])
                        self.train_data.extend(ep_history)
                        if i % 10 == 0 and i != 0:
                            self.train_data = np.array(self.train_data)
                            sess.run(self.nn.update, feed_dict={self.nn.observation: np.vstack(self.train_data[:, 0]), self.nn.reward_holder: self.train_data[:, 1], self.nn.action_holder: self.train_data[:, 2]})
                            self.train_data = []
                        break
                if i % 100 == 0 and i != 0:
                    print(np.mean(self.all_rewards[-100:]))
                    if np.mean(self.all_rewards[-100:]) == self.max_time:
                        break
            self.saver.save(sess, "/tmp/model.ckpt")

agent = Agent()
environment = gym.make("CartPole-v1")
agent.run_agent(environment)