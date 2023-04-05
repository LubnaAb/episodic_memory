import numpy as np


def eval_state_state(state1, state2, semantic_similarity, time_similarity=None):
    """ evaluate the similarity between two states
        state1: np.array of shape (3,)
        state2: np.array of shape (3,)
    """
    semantic1 = state1[0]
    time1 = state1[1]
    episode1 = state1[2]
    semantic2 = state2[0]
    time2 = state2[1]
    episode2 = state2[2]

    semantic_sim = semantic_similarity[int(semantic1), int(semantic2)]
    # if time_similarity is None:
        # time_sim = np.abs(time1 - time2)
    # else:
        # time_sim = time_similarity[int(time1), int(time2)]
    # episode_sim = 1 if episode1 == episode2 else 0
    return semantic_sim 

def eval_state_novel(state, novel_scenario, semantic_similarity, time_similarity=None):
    """ evaluate the similarity between a state and a novel scenario
        state: np.array of shape (3,)
        novel_scenario: np.array of shape (n, 3)
    """
    sim = 0
    for novel_state in novel_scenario:
        sim += eval_state_state(state, novel_state, semantic_similarity, time_similarity)
        return sim / len(novel_scenario)

def eval_sequences_novel(seqs, novel_scenario, semantic_similarity, time_similarity=None):
    """ evaluate the similarity between a sequence of states and a novel scenario
        seqs: np.array of shape (n, 3)
        novel_scenario: np.array of shape (n, 3)
    """
    sim = []
    for seq in seqs:
        sim_i = 0
        for state in seq:
            sim_i += eval_state_novel(state, novel_scenario, semantic_similarity, time_similarity)
        sim.append(sim_i)
    
    mean_sim = np.mean(sim)
    std_sim = np.std(sim)
    return mean_sim, std_sim
