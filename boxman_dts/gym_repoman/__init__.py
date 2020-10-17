from gym.envs.registration import register

register(
    id='RepoMan-v0',
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManBlue-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'blue'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManBeige-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'beige'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManPurple-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'purple'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManPurpleCircle-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'purple' and x.shape == 'circle'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManBeigeSquare-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'beige' and x.shape == 'square'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManBlueSquare-v0',
    kwargs={'goal_condition': lambda x: x.colour == 'blue' and x.shape == 'square'},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManPurpleCircleOrBeigeSquare-v0',
    kwargs={'goal_condition': lambda x: (x.colour == 'purple' and x.shape == 'circle') or (
        x.colour == 'beige' and x.shape == 'square')},
    entry_point='gym_repoman.envs:CollectEnv',
)

register(
    id='RepoManAll-v0',
    kwargs={'termination_condition': lambda collected: len(collected) == 6,
            'reward_condition': lambda _: True},
    entry_point='gym_repoman.envs:MultiCollectEnv',
)
