import argparse
import tensorflow as tf

from controller import Controller


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0', help='OpenAI Gym environment')
    parser.add_argument('--mode', type=str, default='train', help='Train or play')
    parser.add_argument('--scale', type=int, default=10000, help='Scale factor')
    return parser.parse_args()


def main():
    args = parse_args()
    agent = Controller(height=64, width=64, agent_history_length=4, num_filters=[32, 64, 64], filter_shapes=[8, 4, 3],
                       filter_strides=[4, 2, 1], fc_hidden_units=[512], num_actions=4, replay_capacity=100 * args.scale,
                       mini_batch_size=50, target_update_frequency=args.scale, display=False, epsilon_initial=1.0,
                       epsilon_final=0.1, annealing_steps=100 * args.scale, discount_factor=0.99, train_steps=5000 * args.scale,
                       gradient_descent_frequency=4, learning_rate=0.00025, momentum=0.95, rms_decay=0.99,
                       record_loss=True, check_point_steps=100 * args.scale)
    with tf.Session() as sess:
        if args.mode == 'train':
            agent.train(environment_name=args.env, session=sess)
        else:
            agent.play(environment_name=args.env, session=sess, num_steps=10000*args.scale, epsilon=0.05)

if __name__ == "__main__":
    main()
