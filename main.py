import argparse

from scripts.eval import evaluate
from scripts.train import train

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-env", "--environment", type=str,
                           help="Type of environment: spiral, coord, field, direction",
                           choices=["spiral", "coord", "field", "direction"], required=True)
    argParser.add_argument("-a", "--action", type=str, help="Action to be performed, train or eval",
                           required=True, choices=["train", "eval"])
    argParser.add_argument("-pmodel", "--path_model", type=str, help="Filepath in which to save the model for training OR load the model for eval",
                           required=True)
    argParser.add_argument("-pmap", "--path_map", type=str, help="Filepath in which to find the map", required=True)
    argParser.add_argument("-ap", "--agent_placement", type=str, help="Type of placement for the agent: static, random",
                           choices=["static", "random"], default="static")
    argParser.add_argument("-ep", "--enemy_placement", type=str,
                           help="Type of placement for the enemy: static, moves, random",
                           choices=["static", "moves", "random"], default="static")
    argParser.add_argument("-lg", "--logs", type=bool, help="Boolean indicating if logs are required, True or False",
                           choices=[True, False], default=False)
    argParser.add_argument("-llr", "--learn_lr", type=float, help="Learning rate, default is 0.0001", default=0.0001)
    argParser.add_argument("-lts", "--learn_timesteps", type=int,
                           help="Number of timestep to train for, default is 1000000", default=700000)
    argParser.add_argument("-lls", "--learn_learnstart", type=int,
                           help="Number of timestep before training, default is 50000", default=50000)
    argParser.add_argument("-lef", "--learn_exploration", type=float,
                           help="Fraction of total timesteps for which randomness of action is applied", default=0.20)
    argParser.add_argument("-o", "--opti", type=bool,
                           help="Defines if optimization is activated for training and evaluation. /!\ CAREFULL /!\ Available only for the original maps within the project",
                           default=False)
    argParser.add_argument("-mf", "--mode_vision", type=str,
                           help="For the field environment, defines if the vision is static or dynamic",
                           choices=["static", "dynamic"], default="dynamic")
    argParser.add_argument("-meval", "--mode_eval", type=str,
                           help="Type of evaluation, either visual or metric-based",
                           choices=["visual", "metrics"], default="visual")
    argParser.add_argument("-nbep", "--nb_episode", type=int,
                           help="Numer of episodes to evaluate the model on",
                           default=1)
    argParser.add_argument("-v", "--verbose", type=bool,
                           help="Activate the verbose mode or not. Verbose prints each step of the model, non-verbose prints the last one",
                           default=False)
    args = argParser.parse_args()

    if args.action == "train":
        train(args)
    else:
        evaluate(args)
