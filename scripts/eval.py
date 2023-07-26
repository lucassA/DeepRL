import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.utils.env_checker import check_env

from envs.custom_envs.coordFieldVision import CoordFieldVisionEnv
from envs.custom_envs.fullDirectionOnField import FullDirectionOnFieldEnv
from envs.custom_envs.fullFieldVision import FullFieldVisionEnv
from envs.custom_envs.spiralFieldVision import SpiralFieldVisionEnv


def evaluate(env_, policy, neval=10, deterministic=True, mode_eval="metrics", verbose=False):
    if mode_eval == "metrics":
        mean_reward, std_reward = evaluate_policy(model=policy, env=env_, n_eval_episodes=neval,
                                                  deterministic=deterministic)
        print(mean_reward, std_reward)

    elif mode_eval == "visual":
        for i in range(0, neval):
            print("Start:")
            obs, _ = env_.reset()
            env_.render()
            n_steps = 40
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

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-env", "--environment", type=str,
                           help="Type of environment: spiral, coord, field, direction",
                           choices=["spiral", "coord", "field", "direction"], required=True)
    argParser.add_argument("-pmodel", "--path_model", type=str, help="Filepath from which to load the model from",
                           required=True)
    argParser.add_argument("-pmap", "--path_map", type=str, help="Filepath in which to find the map to test on",
                           required=True)
    argParser.add_argument("-ap", "--agent_placement", type=str, help="Type of placement for the agent: static, random",
                           choices=["static", "random"], default="static")
    argParser.add_argument("-ep", "--enemy_placement", type=str,
                           help="Type of placement for the enemy: static, moves, random",
                           choices=["static", "moves", "random"], default="static")
    argParser.add_argument("-meval", "--mode_eval", type=str,
                           help="Type of evaluation, either visual or metric-based",
                           choices=["visual", "metrics"], default="visual")
    argParser.add_argument("-mf", "--mode_vision", type=str,
                           help="For the field environment, defines if the vision is static or dynamic",
                           choices=["static", "dynamic"], default="dynamic")
    argParser.add_argument("-nbep", "--nb_episode", type=int,
                           help="Numer of episodes to evaluate the model on",
                           default=1)
    argParser.add_argument("-v", "--verbose", type=bool,
                           help="Activate the verbose mode or not. Verbose prints each step of the model, non-verbose prints the last one",
                           default=False)
    args = argParser.parse_args()

    env = None
    model = None
    if args.environment == "spiral":
        env = SpiralFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                   player_placement=args.agent_placement,
                                   opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.pmodel, env)
    elif args.environment == "coord":
        env = CoordFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                  player_placement=args.agent_placement,
                                  opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.pmodel, env)
    elif args.environment == "field":
        policy_kwargs = dict(
            features_extractor_kwargs=dict(features_dim=5),
        )
        env = FullFieldVisionEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                 player_placement=args.agent_placement,
                                 opti=args.opti, mode_vision=args.mode_vision)
        check_env(env, warn=False)
        model = DQN.load(args.pmodel, env)
    elif args.environment == "direction":
        env = FullDirectionOnFieldEnv(map_file=args.pmap, enemy_placement=args.enemy_placement,
                                      player_placement=args.agent_placement,
                                      opti=args.opti)
        check_env(env, warn=False)
        model = DQN.load(args.pmodel, env)

    evaluate(env_=env, policy=model, neval=args.nb_episode, mode_eval=args.mode_eval, deterministic=True,
             verbose=args.verbose)
