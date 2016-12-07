import tensorflow as tf
import time
from controller import Controller

start = time.time()
scale = 10000
agent = Controller(height=84, width=84, agent_history_length=4, num_filters=[32, 64, 64], filter_shapes=[8, 4, 3],
                   filter_strides=[4, 2, 1], fc_hidden_units=[512], num_actions=6, replay_capacity=100*scale,
                   mini_batch_size=32, target_update_frequency=scale, display=False, epsilon_initial=1.0,
                   epsilon_final=0.1, annealing_steps=100*scale, discount_factor=0.99, train_steps=5000*scale,
                   clip_size=1.0, gradient_descent_frequency=4, learning_rate=0.00025, momentum=0.95, rms_decay=0.9,
                   record_loss=False, check_point_steps=10*scale)
with tf.Session() as sess:
    agent.train(environment_name='Breakout-v0', session=sess)
    end = time.time()
    agent.play(environment_name='Breakout-v0', session=sess, num_steps=10000*scale, epsilon=0.05)
print 'training time:', end - start
