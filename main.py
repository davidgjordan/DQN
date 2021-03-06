import argparse
import tensorflow as tf

from controller import Controller


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4', help='OpenAI Gym environment')
    parser.add_argument('--num_actions', type=int, default=4, help='Size of action space')
    parser.add_argument('--play', action='store_true', default=False, help='Play')
    parser.add_argument('--screen', type=int, default=84, help='Screen size')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--scale', type=int, default=10000, help='Scale factor')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Log directory')
    return parser.parse_args()


def main():
    args = parse_args()
    agent = Controller(height=args.screen,
                       width=args.screen,
                       agent_history_length=4,
                       num_filters=[32, 64, 64],
                       filter_shapes=[8, 4, 3],
                       filter_strides=[4, 2, 1],
                       fc_hidden_units=[256],
                       num_actions=args.num_actions,
                       replay_capacity=100*args.scale,
                       mini_batch_size=args.batch_size,
                       target_update_frequency=args.scale,
                       display=False,
                       epsilon_initial=1.0,
                       epsilon_final=0.1,
                       annealing_steps=100*args.scale,
                       discount_factor=0.99,
                       train_steps=5000 * args.scale,
                       gradient_descent_frequency=4,
                       learning_rate=0.00025,
                       log_directory=args.log_dir,
                       record_loss=True,
                       check_point_steps=100 * args.scale)
    with tf.Session() as sess:
        if args.play:
            agent.play(environment_name=args.env, session=sess, num_steps=10000 * args.scale, epsilon=0.05)
        else:
            agent.train(environment_name=args.env, session=sess)


if __name__ == "__main__":
    main()
