import os
import argparse
import json
import gym
from agent import TD3


def main(args):
    with open(args.param, "r") as f:
        config = json.load(f)
    config["locexp"] = args.locexp
    path = args.locexp
    # experiment_name = args.experiment_name
    vid_path = os.path.join(path, "videos-{}".format(args.seed))
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    config["vid_path"] = vid_path
    config["res_path"] = res_path
    config["seed"] = args.seed
    print("main seed ", args.seed)
    env = gym.make("LunarLanderContinuous-v2")
    config["max_action"] = env.action_space.high[0]
    config["min_action"] = env.action_space.low[0]
    print(str(config))
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    agent = TD3(action_size=action_size, state_size=state_size, config=config)
    agent.train_agent()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="LunarLanderContinuous-v2", type=str)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--locexp', type=str)
    parser.add_argument('--param', default="param.json", type=str)
    arg = parser.parse_args()
    main(arg)
