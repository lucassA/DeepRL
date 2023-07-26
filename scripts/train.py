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
