import argparse,time,os,sys

import gym

from stable_baselines3 import PPO,A2C,SAC
from stable_baselines3.common.logger import configure

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='',
                        help='Environment: any OpenAI Gym or pyBullet environment may be used')
    # parser.add_argument('--hebb_rule', type=str, default='ABCD_lr', metavar='',
    #                     help='Hebbian rule type: A, AD, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr, ABCD_lr_D_out, ABCD_lr_D_in_and_out')
    # parser.add_argument('--popsize', type=int, default=500, metavar='', help='Population size.')
    # parser.add_argument('--lr', type=float, default=0.2, metavar='', help='ES learning rate.')
    # parser.add_argument('--decay', type=float, default=0.995, metavar='', help='ES learning rate decay.')
    # parser.add_argument('--sigma', type=float, default=0.1, metavar='',
    #                     help='ES sigma: modulates the amount of noise used to populate each new generation')
    # parser.add_argument('--init_weights', type=str, default='uni', metavar='',
    #                     help='The distribution used to sample random weights from at each episode: uni, normal, default, xa_uni, sparse, ka_uni or coevolve to co-evolve the intial weights')
    # parser.add_argument('--print_every', type=int, default=1, metavar='', help='Print and save every N steps.')
    # parser.add_argument('--generations', type=int, default=3000, metavar='',
    #                     help='Number of generations that the ES will run.')
    # parser.add_argument('--threads', type=int, metavar='', default=10,
    #                     help='Number of threads used to run evolution in parallel: -1 uses all threads available')
    # parser.add_argument('--folder', type=str, default='heb_coeffs', metavar='',
    #                     help='folder to store the evolved Hebbian coefficients')
    # parser.add_argument('--distribution', type=str, default='normal', metavar='',
    #                     help='Sampling distribution for initialize the Hebbian coefficients: normal, uniform')

    args = parser.parse_args()


    env = gym.make(args.environment)
    folder_path='./logs/'+'SAC/'+args.environment+'/'+time.asctime( time.localtime(time.time()) ).replace(' ','_')
    new_logger = configure(folder_path, ["stdout", "csv", "tensorboard",'log','json'])
    model = SAC("MlpPolicy", env, verbose=1,train_freq=(1,'episode'),gradient_steps=-1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=10000000)


if __name__ == '__main__':
    main(sys.argv)
