from gym.envs.registration import register

register(
    id='roar-v0',
    entry_point='ROAR_Gym.envs:ROAREnv',
)

register(
    id='roar-pid-v0',
    entry_point='ROAR_Gym.envs:ROARPIDEnv',
)
register(
    id='roar-local-planner-v0',
    entry_point='ROAR_Gym.envs:LocalPlannerEnv'
)
register(
    id='roar-local-planner-v1',
    entry_point='ROAR_Gym.envs:LocalPlannerEnv1'
)
register(
    id='roar-depth-e2e-v0',
    entry_point='ROAR_Gym.envs:DepthE2EEnv'
)
register(
    id='roar-occu-map-e2e-v0',
    entry_point='ROAR_Gym.envs:OccuMapE2EEnv'
)
register(
    id='roar-occu-debug-v0',
    entry_point='ROAR_Gym.envs:OccuDebug'
)