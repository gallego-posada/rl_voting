import random
from collections import Counter
import numpy as np

def scores_to_profile(S):
    n, m = S.shape
    return np.array([np.argsort(-S[i,:]) for i in range(n)])

def get_winner(ctr, m):

    # Sort candidates according to their score (uses lexicographic ordering)
    sorted_cands = ctr.most_common(m)

    # Take the score of the first sorted candidate
    max_score = sorted_cands[0][1]

    # Create set of candidates with maximum score
    winner_set = [sorted_cands[cand][0] for cand in range(len(sorted_cands)) if sorted_cands[cand][1] == max_score]

    # Break tie among winner_set randomly
    return random.sample(winner_set, 1)[0]

def plurality(S):

    n, m = S.shape

    candidate_list = [np.argmax(S[i,:]) for i in range(n)]

    # Count number of time each alternative is ranked first
    ctr = Counter(candidate_list)

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
    return voting_rule(S)
