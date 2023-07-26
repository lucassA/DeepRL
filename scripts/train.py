import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from gymnasium.utils.env_checker import check_env

from envs.custom_envs.coordFieldVision import CoordFieldVisionEnv
from envs.custom_envs.fullDirectionOnField import FullDirectionOnFieldEnv
from envs.custom_envs.fullFieldVision import FullFieldVisionEnv
from envs.custom_envs.spiralFieldVision import SpiralFieldVisionEnv


def train(args):
    model = None
    if args.environment == "spiral":
        env = SpiralFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                   player_placement=args.agent_placement,
                                   opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.learn_learnstart, train_freq=4,
                    exploration_fraction=args.learn_exploration, learning_rate=args.learn_lr)
    elif args.environment == "coord":
        env = CoordFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                  player_placement=args.agent_placement,
                                  opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.learn_learnstart, train_freq=4,
                    exploration_fraction=args.learn_exploration, learning_rate=args.learn_lr)
    elif args.environment == "field":
        policy_kwargs = dict(
            features_extractor_kwargs=dict(features_dim=5),
        )
        env = FullFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                 player_placement=args.agent_placement,
                                 opti=args.opti, mode_vision=args.mode_vision)
        check_env(env, warn=False)
        model = DQN("CnnPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.learn_learnstart, train_freq=4,
                    exploration_fraction=args.learn_exploration, learning_rate=args.learn_lr, policy_kwargs=policy_kwargs)
    elif args.environment == "direction":
        env = FullDirectionOnFieldEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                      player_placement=args.agent_placement,
                                      opti=args.opti)
        check_env(env, warn=False)
        model = DQN("MlpPolicy", env, buffer_size=1000000, verbose=1, learning_starts=args.learn_learnstart, train_freq=4,
                    exploration_fraction=args.learn_exploration, learning_rate=args.learn_lr)

    # Logger
    if args.logs:
        logs_path = "logs"
        new_logger = configure(logs_path, ["csv"])
        model.set_logger(new_logger)

    # Train the agent
    print("Training starts")
    model.learn(total_timesteps=args.learn_timesteps)
    print("Training ends")
    print("Saving the model at:",args.path_model)
    model.save(args.path_model)
