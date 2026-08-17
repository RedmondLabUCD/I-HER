import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    
    parser.add_argument('--dynamics-steps', type=int, default=1e4, help='grad steps per dynamics update')
    parser.add_argument('--real-rollouts', type=int, default=256, help='real rollouts to collect per epoch')
    parser.add_argument('--dynamics-hiddens', type=int, default=256, help='num hidden layers in dynamics model')
    parser.add_argument('--exp-name', type=str, default='temp_exp', help='name of the experiment')
    parser.add_argument('--simulator', type=str, default='mujoco', help='which simulator to use')
    parser.add_argument('--save-models', type=int, default=1, help='whether to save model weights each epoch')
    parser.add_argument('--save-every', type=int, default=10, help='frequency to save ac model')
    parser.add_argument('--bias-real', type=int, default=1, help='How much to bias toward real experience when sampling for policy updates')
    parser.add_argument('--bias-recent-refill', type=int, default=1, help='How much to bias toward more recent policies when refilling imag buffer')
    parser.add_argument('--bias-recent-sample', type=int, default=2, help='How much to bias toward more recent experience when sampling to update dynamics')
    parser.add_argument('--include-ri', type=int, default=1, help='whether to include intrinsic rewards')
    parser.add_argument('--render', type=int, default=0, help='whether to render environment')
    parser.add_argument('--p-realwreal', type=float, default=0.5, help='probability of collecting real rollout with real controller')
    parser.add_argument('--p-imagwreal', type=float, default=0.1, help='probability of collecting imag rollout with real controller')
    parser.add_argument('--epochs-per-dy-update', type=int, default=5, help='How often to update dynamics model')
    parser.add_argument('--distinguish', type=int, default=1, help='Whether to distinguish real and imaginary experiences')
    parser.add_argument('--refill', type=int, default=1, help='Whether to refill imag buffer after dynamics model is updated')
    parser.add_argument('--squared-l2-loss', type=int, default=1, help='Whether to keep dynamics model loss squared')
    parser.add_argument('--ensemble-size', type=int, default=5, help='Size of dynamics model ensemble')
    parser.add_argument('--dynamics-batch-size', type=int, default=512, help='Dynamics model batch size')
    
    parser.add_argument('--ensemble-path', type=str, default='pretrained/ensemble', help='the path to the pretrained ensemble of dynamics models')
    parser.add_argument('--ac-path', type=str, default='pretrained/ac', help='the path to the pretrained actor-critic model')
    parser.add_argument('--load-ac', type=int, default=0, help='Whether to load in pretrained actor critic')
    

    args = parser.parse_args()

    return args
