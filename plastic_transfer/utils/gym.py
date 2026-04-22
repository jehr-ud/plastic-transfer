import numpy as np

def get_obs_dim(space):
    if hasattr(space, "spaces"):  # Dict
        return sum(get_obs_dim(s) for s in space.spaces.values())
    else:  # Box
        return int(np.prod(space.shape))
    

def get_action_dim(space):
    return int(np.prod(space.shape))