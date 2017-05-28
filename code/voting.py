import random
from collections import Counter
import numpy as np

def scores_to_profile(S):
    n, m = S.shape
    # print(np.array([np.argsort(-S[i,:]) for i in range(n)]))
    return np.array([np.argsort(-S[i,:]) for i in range(n)])

def get_winner(ctr, m):

    # Sort candidates according to their score (uses lexicographic ordering)
    sorted_cands = ctr.most_common(m)

    # Add all those candidates for which we dont have a score, with score 0
    for cand in set(range(m)) - set([pair[0] for pair in sorted_cands]):
        sorted_cands.append((cand, 0))

    #print(sorted_cands)

    levels = sorted(set([pair[1] for pair in sorted_cands]))
    final_list = []
    for level in levels:
        level_set = [pair[0] for pair in sorted_cands if pair[1] == level]
        random.shuffle(level_set)
        final_list += level_set

        # level_sets[i][1]


    final_list = final_list[::-1]

    #print(final_list)

    return final_list

    # # Take the score of the first sorted candidate
    # max_score = sorted_cands[0][1]
    #
    # # Create set of candidates with maximum score
    # winner_set = [sorted_cands[cand][0] for cand in range(len(sorted_cands)) if sorted_cands[cand][1] == max_score]

    # Break tie among winner_set randomly
    #return [sorted_cands[cand][0] for cand in range(len(sorted_cands))]
    #return random.sample(winner_set, 1)[0]

def plurality(S):

    n, m = S.shape

    profile = scores_to_profile(S)

    # Count number of time each alternative is ranked first
    ctr = Counter(profile[:,0])

    return get_winner(ctr, m)

def borda(S):
    n,m = S.shape

    profile = scores_to_profile(S)

    res = np.zeros(m)

    # Loop over the ballots
    for i in range(n):
        for k in range(m):
            # Increase the score for the candidate according to the Borda score
            res[profile[i,k]] += (m-1) - k

    score_dict = {}
    for k in range(m):
        score_dict[k] = res[k]

    # Count number of Borda points
    ctr = Counter(score_dict)

    return get_winner(ctr, m)

def hundred_points(S):
    n,m = S.shape

    total_score = np.sum(S, axis = 0)

    score_dict = {}
    for k in range(m):
        score_dict[k] = total_score[k]

    # Accumulate scores for each candidate
    ctr = Counter(score_dict)

    return get_winner(ctr, m)

def copeland(S):
    n,m = S.shape

    profile = scores_to_profile(S)

    vote_graph = np.zeros((m,m))
    for voter in range(n):
        for k in range(m):
            winner = profile[voter,k]
            for k_ in range(k+1, m):
                loser = profile[voter,k_]
                vote_graph[winner, loser] += 1

    # Turn vote_graph into a majority graph, and calculate Copeland score
    res = np.sum(1. * (vote_graph - vote_graph.T > 0), axis = 1)

    score_dict = {}
    for k in range(m):
        score_dict[k] = res[k]

    # Accumulate scores for each candidate
    ctr = Counter(score_dict)

    return get_winner(ctr, m)

def vote(S, voting_rule):
    # print("8*******************")
    # print(S)
    # print("---------------")
    return voting_rule(S)
