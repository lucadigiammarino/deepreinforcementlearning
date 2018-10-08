"""
Reinforcement Learning Project - Section AI 2.B
author: Luca Di Giammarino
date: 07/03/2018
"""


import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

# MDP syntax used
# action -> a
# state -> x
# next state -> next_x
# reward -> r

if "../" not in sys.path:
  sys.path.append("../")

from collections import deque, namedtuple

env = gym.envs.make("Gopher-v0")

# Atari Actions for Gopher:
VALID_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7]

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            # obtain input image
            self.input_x = tf.placeholder(shape=[250, 160, 3], dtype=tf.uint8)
            # convert to greyscale
            self.output = tf.image.rgb_to_grayscale(self.input_x)
            # crop image
            self.output = tf.image.crop_to_bounding_box(self.output, 110, 0, 120, 160)
            # resize image
            self.output = tf.image.resize_images(self.output, [50, 50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # removes all dimensions of size 1
            self.output = tf.squeeze(self.output)

    def process(self, sess, x):
        """
            :param sess: tensorflow session object
            :param x: [210, 160, 3] atari RGB State
            :return: processed [84, 84, 1] x representing grayscale values
        """
        return sess.run(self.output, { self.input_x: x })

class Approximator():
    """Q-Value Approximator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="approximator", summaries_dir=None, sess=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                # create directory if does not exist
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    
    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        
        # Our input are 4 RGB frames of shape 50, 50 each
        self.X_pl = tf.placeholder(shape=[None, 50, 50, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which awas selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]
        
        # three convolutional layers
        with tf.name_scope('Convolutional_Layer_1') as scope:
            conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        with tf.name_scope('Convolutional_Layer_2') as scope:
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        with tf.name_scope('Convolutional_Layer_3') as scope:
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # flattened layer
        with tf.name_scope('Flattened_Layer') as scope:
            flattened = tf.contrib.layers.flatten(conv3)

        # fully connected layer
        with tf.name_scope('Fully_Connected_Layer') as scope:
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # get the predictions for the choses actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        # slices from predictions according to indices
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # calculate the losses using the square difference (x-y)(x-y)
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)

        # reduce input losses, return tensor with single element
        with tf.name_scope('Loss') as scope:
            self.loss = tf.reduce_mean(self.losses)

        # optimizer parameters from original paper
        with tf.name_scope('Optimization') as scope:
            # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.optimizer = tf.train.AdamOptimizer(1e-4)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # summaries for Tensorboard
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Max_Q_Value", tf.reduce_max(self.predictions))
        tf.summary.histogram("Loss_Hist", self.losses)
        tf.summary.histogram("Q_Values_Hist", self.predictions)
            
        self.summaries = tf.summary.merge_all()


    def predict(self, sess, s):
        """
        Predicts avalues.

        :param sess: Tensorflow session
        :param s: State input of shape [batch_size, 4, 160, 160, 3]

        :return Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated avalues.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Predict avalues
        :param self:
        :param sess: tensorflow session object
        :param s: x input of shape [batch_size, 4, 160, 160, 3]
        :param a: chosen actions of shape [batch_size]
        :param y: targets of shape [batch_size]
        :return: the calculated loss on the batch
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

def copy_model_parameters(sess, approximator1, approximator2):
    """
    Copies the model parameters of one approximator to another
    
    :param sess: Tensorflow session instance
    :param approximator1: approximator to copy the paramters from
    :param approximator2: approximator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(approximator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(approximator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(approximator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon
    :param approximator: An approximator that returns q values for a given x
    :param nA: Number of actions in the environment.
    
    :return a function that takes the (sess, observation, epsilon) as an argument and returns
    the probabilities for each ain the form of a numpy array of length nA.
    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = approximator.predict(sess, np.expand_dims(observation, 0))[0]
        best_a= np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def deep_Q_learn(sess,
                    env,
                    q_approximator,
                    target_approximator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_approximator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    :param sess: Tensorflow Session object
    :param env: OpenAI environment
    :param q_approximator: Approximator object used for the q values
    :param target_approximator: Approximator object used for the targets
    :param state_processor: A StateProcessor object
    :param num_episodes: Number of episodes to run for
    :param experiment_dir: Directory to save Tensorflow summaries in
    :param replay_memory_size: Size of the replay memory
    :param replay_memory_init_size: Number of random experiences to sampel when initializing the reply memory.
    :param update_target_approximator_every: Copy parameters from the Q approximator to the target approximator every N steps
    :param discount_factor: Gamma discount factor
    :param epsilon_start: Chance to sample a random awhen taking an action. Epsilon is decayed over time and this is the start value
    :param epsilon_end: The final minimum value of epsilon after decaying is done
    :param epsilon_decay_steps: Number of steps to decay epsilon over
    :param batch_size: Size of batches to sample from the replay memory
    :param record_video_every: Record a video every N episodes

    :return An EpisodeStats object with two numpy arrays for episode_lengths and episode_rs.
    """

    Transition = namedtuple("Transition", ["x", "a", "r", "next_x", "done"])

    # the replay memory
    replay_memory = []


    # create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # the epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # the policy we're following
    policy = make_epsilon_greedy_policy(
        q_approximator,
        len(VALID_ACTIONS))

    # populate the replay memory with initial experience
    print("Populating replay memory...")
    x = env.reset()
    x = state_processor.process(sess, x)
    x = np.stack([x] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, x, epsilons[min(total_t, epsilon_decay_steps-1)])
        a = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_x, r, done, _ = env.step(VALID_ACTIONS[action])
        next_x = state_processor.process(sess, next_x)
        next_x = np.append(x[:,:,1:], np.expand_dims(next_x, 2), axis=2)
        replay_memory.append(Transition(x, action, r, next_x, done))
        if done:
            x = env.reset()
            x = state_processor.process(sess, x)
            x = np.stack([x] * 4, axis=2)
        else:
            x = next_x

    # record videos use the gym env Monitor wrapper
    env = Monitor(env, directory=monitor_path, resume=True, video_callable=lambda count: count % record_video_every ==0)

    for i_episode in range(num_episodes):

        # save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # reset the environment
        x = env.reset()
        x = state_processor.process(sess, x)
        x = np.stack([x] * 4, axis=2)
        loss = None

        # one step in the environment
        for t in itertools.count():

            # epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_approximator.summary_writer.add_summary(episode_summary, total_t)

            # maybe update the target approximator
            if total_t % update_target_approximator_every == 0:
                copy_model_parameters(sess, q_approximator, target_approximator)
                print("\nCopied model parameters to target network.")

            # print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # take a step
            action_probs = policy(sess, x, epsilon)
            a = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_x, r, done, _ = env.step(VALID_ACTIONS[action])
            next_x = state_processor.process(sess, next_x)
            next_x = np.append(x[:,:,1:], np.expand_dims(next_x, 2), axis=2)

            # if our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # save transition to replay memory
            replay_memory.append(Transition(x, action, r, next_x, done))

            # update statistics
            statistics.episode_rs[i_episode] += r
            statistics.episode_lengths[i_episode] = t

            # sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            xs_batch, action_batch, r_batch, next_xs_batch, done_batch = map(np.array, zip(*samples))

            # calculate q values and targets (Double DQN)
            q_values_next = q_approximator.predict(sess, next_xs_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_approximator.predict(sess, next_xs_batch)
            targets_batch = r_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # perform gradient descent update
            xs_batch = np.array(xs_batch)
            loss = q_approximator.update(sess, xs_batch, action_batch, targets_batch)

            if done:
                break

            x = next_x
            total_t += 1

        # add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=statistics.episode_rs[i_episode], node_name="episode_reward", tag="Episode_Reward")
        episode_summary.value.add(simple_value=statistics.episode_lengths[i_episode], node_name="episode_length", tag="Episode_Length")
        q_approximator.summary_writer.add_summary(episode_summary, total_t)
        q_approximator.summary_writer.flush()

    env.monitor.close()
    return statistics


# where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    
    # create approximators
    q_approximator = Approximator(scope="q", summaries_dir=experiment_dir, sess=sess)
    target_approximator = Approximator(scope="target_q")
    
    # x processor
    state_processor = StateProcessor()
    sess.run(tf.global_variables_initializer())
    for t, statistics in deep_Q_learn(sess,
                                    env,
                                    q_approximator=q_approximator,
                                    target_approximator=target_approximator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_approximator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(statistics.episode_rs[-1]))

