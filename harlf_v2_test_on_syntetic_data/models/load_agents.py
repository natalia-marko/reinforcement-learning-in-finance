def load_base_agents():
    """Helper function to load base agents"""
    from stable_baselines3 import PPO, SAC, DDPG, TD3
    import pickle

    # Load agent names
    with open('models/base_agent_names.pkl', 'rb') as f:
        agent_names = pickle.load(f)

    # Load each agent
    loaded_agents = {}
    for agent_name in agent_names:
        # Determine agent type
        if 'PPO' in agent_name:
            agent = PPO.load(f'models/base_agent_{agent_name}')
        elif 'SAC' in agent_name:
            agent = SAC.load(f'models/base_agent_{agent_name}')
        elif 'DDPG' in agent_name:
            agent = DDPG.load(f'models/base_agent_{agent_name}')
        elif 'TD3' in agent_name:
            agent = TD3.load(f'models/base_agent_{agent_name}')

        loaded_agents[agent_name] = agent

    return loaded_agents
