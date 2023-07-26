import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.utils.env_checker import check_env

from envs.custom_envs.coordFieldVision import CoordFieldVisionEnv
from envs.custom_envs.fullDirectionOnField import FullDirectionOnFieldEnv
from envs.custom_envs.fullFieldVision import FullFieldVisionEnv
from envs.custom_envs.spiralFieldVision import SpiralFieldVisionEnv


def perform_evaluation(env_, policy, neval=10, deterministic=True, mode_eval="metrics", verbose=False):
    if mode_eval == "metrics":
        mean_reward, std_reward = evaluate_policy(model=policy, env=env_, n_eval_episodes=neval,
                                                  deterministic=deterministic)
        print(mean_reward, std_reward)

    elif mode_eval == "visual":
        for i in range(0, neval):
            print("Start:")
            obs, _ = env_.reset()
            env_.render()
            n_steps = 50
            for step in range(n_steps):
                action, _ = policy.predict(obs, deterministic=deterministic)
                obs, reward, done, terminate, info = env_.step(action)
                if verbose:
                    print(f"Step {step + 1}")
                    print("Action: ", action)
                    print("reward=", reward)
                    env_.render()
                if done:
                    print("Goal reached!", "reward=", reward)

                    break
            print("Finish:")
            env_.render()
    else:
        print("mode_eval not recognized")


#################

def evaluate(args):
    env = None
    model = None
    if args.environment == "spiral":
        env = SpiralFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                   player_placement=args.agent_placement,
                                   opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.path_model, env)
    elif args.environment == "coord":
        env = CoordFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                  player_placement=args.agent_placement,
                                  opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.path_model, env)
    elif args.environment == "field":
        env = FullFieldVisionEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                 player_placement=args.agent_placement,
                                 opti=args.opti, mode_vision=args.mode_vision)
        check_env(env, warn=False)
        model = DQN.load(args.path_model, env)
    elif args.environment == "direction":
        env = FullDirectionOnFieldEnv(map_file=args.path_map, enemy_placement=args.enemy_placement,
                                      player_placement=args.agent_placement,
                                      opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.path_model, env)

    perform_evaluation(env_=env, policy=model, neval=args.nb_episode, mode_eval=args.mode_eval, deterministic=True,
             verbose=args.verbose)
