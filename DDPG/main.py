
import argparse
import json
import gym
from agent import DDPGAgent


def main(args):
    with open ("param.json", "r") as f:
        config = json.load(f)
    config["seed"] = args.seed
    env = gym.make("LunarLanderContinuous-v2")
    config["max_action"] = env.action_space.high[0]
    config["min_action"] = env.action_space.low[0]
    print(str(config))
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    agent = DDPGAgent(action_size=action_size, state_size=state_size, config=config)
    agent.train_agent()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="LunarLanderContinuous-v2", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg = parser.parse_args()
    main(arg)
