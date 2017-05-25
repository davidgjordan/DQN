import tensorflow as tf
import gym
from memory import ReplayMemory
from state import State
from CNN import ConvNet
import random
import numpy as np


class Controller(object):
    def __init__(self, height, width, agent_history_length, num_filters, filter_shapes, filter_strides,
                 fc_hidden_units, num_actions, replay_capacity, mini_batch_size, target_update_frequency, display,
                 train_steps, epsilon_initial, epsilon_final, annealing_steps, learning_rate, discount_factor,
                 gradient_descent_frequency, momentum, rms_decay, record_loss, check_point_steps):

        self.height = height
        self.width = width
        self.agent_history_length = agent_history_length
        self.replay_capacity = replay_capacity
        self.train_steps = train_steps
        self.display = display
        self.num_actions = num_actions
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.annealing_steps = annealing_steps
        self.epsilon_slope = (epsilon_final - epsilon_initial) / annealing_steps
        self.discount_factor = discount_factor
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.rms_decay = rms_decay
        self.momentum = momentum
        self.gradient_descent_frequency = gradient_descent_frequency
        self.target_update_frequency = target_update_frequency
        self.record_loss = record_loss
        self.check_point_steps = check_point_steps

        self.target_net, self.predictor_net = [ConvNet(height=height, width=width, num_channels=agent_history_length,
                                                       num_filters=num_filters, filter_shapes=filter_shapes,
                                                       filter_strides=filter_strides, fc_hidden_units=fc_hidden_units,
                                                       num_class=num_actions) for _ in range(2)]

        self.update_target_op = [t_w.assign(p_w) for t_w, p_w in zip(self.target_net.weights, self.predictor_net.weights)]

        self.state, self.new_state = [State(height=self.height,
                                            width=self.width,
                                            depth=self.agent_history_length) for _ in range(2)]

        self.actions, self.rewards, self.terminals = self.create_placeholders()
        self.loss = self.create_scalar_loss()
        self.train_step = self.create_train_step()
        self.memory = ReplayMemory(replay_capacity, mini_batch_size)
        self.writer = None
        self.merged = None
        self.environment = None
        self.session = None
        self.saver = tf.train.Saver()
        self.score = 0
        self.episode = 0

    def train(self, environment_name, session):
        self.prepare_controller(environment_name, session)
        self.episode = 0
        print 'Training by minibatch gradient descent...'
        for t in xrange(self.train_steps):
            epsilon = self.annealed_epsilon(t)
            self.follow_policy(epsilon)
            self.gradient_descent(t)
            self.save_weights(t)

    def follow_policy(self, epsilon):
        action = self.choose_action(epsilon)
        reward, terminal = self.observe_reward(action)
        self.record_experience(action, reward, terminal)

    def choose_action(self, epsilon):
        if random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            Q_values = self.session.run(self.predictor_net.hypothesis,
                                        feed_dict={self.predictor_net.x: [self.state.content/255.0]})
            return np.argmax(Q_values)

    def observe_reward(self, action):
        self.state.content = np.copy(self.new_state.content)
        observation, reward, terminal, info = self.environment.step(action)
        self.score += reward
        self.new_state.update(observation)
        return reward, terminal

    def record_experience(self, action, reward, terminal):
        experience = (np.copy(self.state.content), action, reward, np.copy(self.new_state.content), terminal)
        self.memory.insert(experience)
        if terminal:
            self.calculate_average_reward()
            observation = self.environment.reset()
            self.new_state.update(observation)

    # updates weights of predictor network using a random minibatch of experience from the memory
    def gradient_descent(self, time_step):
        if time_step % self.gradient_descent_frequency is not 0:
            return
        replay_batch = self.memory.draw()
        feed_dict = self.create_feed_dict(replay_batch)
        self.session.run(self.train_step, feed_dict=feed_dict)
        if self.record_loss and time_step % 10000 is 0:
            summary, loss = self.session.run([self.merged, self.loss], feed_dict=feed_dict)
            print time_step, loss
            self.writer.add_summary(summary, time_step)

    def annealed_epsilon(self, time_step):
        # run training checks
        if self.display:
            self.environment.render()
        if time_step % self.target_update_frequency is 0:
            self.update_target_net()
        # decay epsilon on a schedule
        return self.epsilon_final if time_step > self.annealing_steps else \
            time_step * self.epsilon_slope + self.epsilon_initial

    def save_weights(self, time_step):
        if time_step % self.check_point_steps is 0:
            self.saver.save(self.session, 'logs/checkpoints' + 'model.ckpt', global_step=time_step)
            print time_step, 'checkpoint reached'

    def initialize_memory(self):
        print 'Populating replay memory with random experiences...'
        self.environment.reset()
        t = 1
        while self.memory.len() < self.replay_capacity:
            self.move_randomly(t)
            t += 1

    def move_randomly(self, time_step):
        if self.display:
            self.environment.render()
        action = np.random.randint(self.num_actions)
        reward, terminal = self.observe_reward(action)
        if time_step > self.agent_history_length:
            self.record_experience(action, reward, terminal)

    def play(self, environment_name, session, epsilon, num_steps):
        ckpt = tf.train.get_checkpoint_state('logs')
        self.saver.restore(session, ckpt.model_checkpoint_path)
        self.session = session
        self.environment = gym.make(environment_name)
        self.environment.monitor.start('logs/gym', force=True, video_callable=lambda x: True)
        self.environment.reset()
        print 'Playing the game...'
        for t in xrange(num_steps):
            self.environment.render()
            action = self.choose_action(epsilon)
            reward, terminal = self.observe_reward(action)
            if terminal:
                self.environment.reset()
        self.environment.monitor.end()

    # Helper methods
    def prepare_controller(self, environment_name, session):
        self.session = session
        self.writer = tf.summary.FileWriter('logs/Controller', session.graph)
        self.merged = tf.summary.merge_all()
        self.environment = gym.make(environment_name)
        self.initialize_memory()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def create_scalar_loss(self):
        with tf.name_scope('loss'):
            Q_target = self.create_target()
            Q_train = self.create_train()
            with tf.name_scope('average_clipped_error'):
                average_clipped_error = tf.reduce_mean(clipped_error(Q_target - Q_train))
            tf.summary.scalar("loss", average_clipped_error)
        return average_clipped_error

    def create_target(self):
        with tf.name_scope('Q_target'):
            Q_max = tf.reduce_max(self.target_net.hypothesis, axis=1, name='Q_max')
            with tf.name_scope('check_terminal_condition'):
                Q_target = tf.where(self.terminals, self.rewards, self.discount_factor * Q_max + self.rewards)
        return Q_target

    def create_train(self):
        with tf.name_scope('Q_train'):
            one_hot_actions = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name='one_hot_actions')
            Q_train = tf.reduce_sum(tf.multiply(self.predictor_net.hypothesis, one_hot_actions), axis=1)
        return Q_train

    def create_feed_dict(self, batch):
        batch_states = np.array([batch[i][0] for i in range(self.mini_batch_size)]).astype(np.float32)/255
        batch_actions = np.array([batch[i][1] for i in range(self.mini_batch_size)])
        batch_rewards = np.array([batch[i][2] for i in range(self.mini_batch_size)]).astype(np.float32)
        batch_next_states = np.array([batch[i][3] for i in range(self.mini_batch_size)]).astype(np.float32)/255
        batch_terminals = np.array([batch[i][4] for i in range(self.mini_batch_size)])
        return {self.predictor_net.x: batch_states,
                self.target_net.x: batch_next_states,
                self.actions: batch_actions,
                self.rewards: batch_rewards,
                self.terminals: batch_terminals}

    def create_train_step(self):
        with tf.name_scope('train'):
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                             decay=self.rms_decay,
                                             momentum=self.momentum,
                                             epsilon=1e-6). \
                minimize(self.loss, var_list=self.predictor_net.weights)

    def update_target_net(self):
        self.session.run(self.update_target_op)

    def calculate_average_reward(self):
        self.episode += 1
        if self.episode % 100 is 0:
            print 'reward per 100 episodes', self.score / self.episode
            self.score = 0
            self.episode = 0

    @staticmethod
    def create_placeholders():
        with tf.name_scope("action-reward-terminal-inputs"):
            action_ph = tf.placeholder(tf.int32, [None], name='action')
            reward_ph = tf.placeholder(tf.float32, [None], name='reward')
            terminal_ph = tf.placeholder(tf.bool, [None], name='terminal')
        return action_ph, reward_ph, terminal_ph

    @staticmethod
    def clipped_error(x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
