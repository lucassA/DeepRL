import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from gymnasium.utils.env_checker import check_env

from envs.custom_envs.coordFieldVision import CoordFieldVisionEnv
from envs.custom_envs.fullDirectionOnField import FullDirectionOnFieldEnv
from envs.custom_envs.fullFieldVision import FullFieldVisionEnv
from envs.custom_envs.spiralFieldVision import SpiralFieldVisionEnv

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-env", "--environment", type=str,
                           help="Type of environment: spiral, coord, field, direction",
                           choices=["spiral", "coord", "field", "direction"], required=True)
    argParser.add_argument("-pmodel", "--path_model", type=str, help="Filepath in which to save the model",
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
                           help="Number of timestep to train for, default is 1000000", default=1000000)
    argParser.add_argument("-lls", "--learn_learnstart", type=int,
                           help="Number of timestep before training, default is 50000", default=50000)
    argParser.add_argument("-lef", "--learn_exploration", type=float,
                           help="Fraction of total timesteps for which randomness of action is applied", default=0.20)
    argParser.add_argument("-o", "--opti", type=bool,
                           help="Defines if optimization is activated for training. Available only for the original maps within the project",
                           default=True)
    argParser.add_argument("-mf", "--mode_vision", type=str,
                           help="For the field environment, defines if the vision is static or dynamic",
                           choices=["static", "dynamic"], default="dynamic")
    args = argParser.parse_args()

    env = None
    model = None
    if args.environment == "spiral":
        env = SpiralFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                   player_placement=args.agent_placement,
                                   opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.lls, train_freq=4,
                    exploration_fraction=args.lef, learning_rate=args.llr)
    elif args.environment == "coord":
        env = CoordFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                  player_placement=args.agent_placement,
                                  opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.lls, train_freq=4,
                    exploration_fraction=args.lef, learning_rate=args.llr)
    elif args.environment == "field":
        policy_kwargs = dict(
            features_extractor_kwargs=dict(features_dim=5),
        )
        env = FullFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                 player_placement=args.agent_placement,
                                 opti=args.opti, mode_vision=args.mode_vision)
        check_env(env, warn=False)
        model = DQN("CnnPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.lls, train_freq=4,
                    exploration_fraction=args.lef, learning_rate=args.llr, policy_kwargs=policy_kwargs)
    elif args.environment == "direction":
        env = FullDirectionOnFieldEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                      player_placement=args.agent_placement,
                                      opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.lls, train_freq=4,
                    exploration_fraction=args.lef, learning_rate=args.llr)





    # Logger
    if args.logs:
        logs_path = "logs"
        new_logger = configure(logs_path, ["csv"])
        model.set_logger(new_logger)

    # Train the agent
    print("Training starts")
    model.learn(total_timesteps=args.lts)
    print("Training ends")
    print("Saving the model at:",args.pmodel)
    model.save(args.pmodel)
