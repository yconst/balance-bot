### Balance Bot

This is an environment for [OpenAI Gym](https://github.com/openai/gym) where the goal is to train a controller for a two-wheeled balancing robot. The aim is to stay upright as long as possible, and maintain desired speed (by default zero, ie stationary).

To install this environment:

    git clone https://github.com/yconst/balance-bot
    cd balance-bot
    pip install -e .
    
A basic script for training a Deep-Q agent:

    import gym
    from baselines import deepq
    import balance_bot

    def callback(lcl, glb):
        # stop training if reward exceeds 199
        is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
        return is_solved

    def main():
        env = gym.make("balancebot-v0")
        model = deepq.models.mlp([16, 12])
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=100000,
            buffer_size=100000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=callback
        )
        print("Saving model to balance.pkl")
        act.save("balance.pkl")

    if __name__ == '__main__':
        main()

For more information on how to setup and train a model take a look at [this blog post](https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/).

Released under [MIT License](https://opensource.org/licenses/MIT).
