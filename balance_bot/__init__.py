import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='balancebot-v0',
    entry_point='balance_bot.envs:BalancebotEnv',
)

register(
    id='balancebot-noise-v0',
    entry_point='balance_bot.envs:BalancebotEnvNoise',
)
