from __future__ import print_function
import gym
from builtins import range
import time
import numpy as np
from ..envs.utils import getSpaceInfo
from joblib import Parallel, delayed


def _eval_and_render(mdp, policy, nbEpisodes=1, metric='discounted',
                     initialState=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        stepConfidence (float):  95% confidence level for step average
    """
    fps = mdp.metadata.get('video.frames_per_second') or 100
    values = np.zeros(nbEpisodes)
    steps = np.zeros(nbEpisodes)
    gamma = mdp.gamma
    if metric == 'average':
        gamma = 1
    for e in range(nbEpisodes):
        epPerformance = 0.0
        df = 1
        t = 0
        H = None
        done = False
        if render:
            mdp.render(mode='human')
        if hasattr(mdp, 'horizon'):
            H = mdp.horizon
        mdp.reset()
        state = mdp._reset(initialState)
        while (t < H) and (not done):
            action = policy.drawAction(state)
            state, r, done, _ = mdp.step(action)
            epPerformance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        # if(t>= H):
        #    print("Horizon!!")
        if gamma == 1:
            epPerformance /= t
        print("\tperformance", epPerformance)
        values[e] = epPerformance
        steps[e] = t

    return values.mean(), 2 * values.std() / np.sqrt(nbEpisodes), steps.mean(), 2 * steps.std() / np.sqrt(nbEpisodes)


def _eval_and_render_vectorial(mdp, policy, nbEpisodes=1, metric='discounted',
                               initialState=None, render=True):
    """
    This function evaluate a policy on the specified metric by executing
    multiple episode and visualize its performance
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
        step (float): average number of step before finish
        stepConfidence (float):  95% confidence level for step average
    """
    fps = mdp.metadata.get('video.frames_per_second') or 100
    values = np.zeros(nbEpisodes)
    steps = np.zeros(nbEpisodes)
    gamma = mdp.gamma
    if metric == 'average':
        gamma = 1
    for e in range(nbEpisodes):
        epPerformance = 0.0
        df = 1
        t = 0
        H = None
        done = False
        if render:
            mdp.render(mode='human')
        if hasattr(mdp, 'horizon'):
            H = mdp.horizon
        mdp.reset()
        state = mdp._reset(initialState)
        while (t < H) and (not done):
            action = policy.drawAction(state)
            state, r, done, _ = mdp.step(action)
            epPerformance += df * r
            df *= gamma
            t += 1

            if render:
                mdp.render()
                time.sleep(1.0 / fps)
        # if(t>= H):
        #    print("Horizon!!")
        if gamma == 1:
            epPerformance /= t
        print("\tperformance", epPerformance)
        values[e] = epPerformance
        steps[e] = t

    return values, steps


def _parallel_eval(mdp, policy, nbEpisodes, metric, initialState, n_jobs, nEpisodesPerJob):
    # TODO using joblib
    # return _eval_and_render(mdp, policy, nbEpisodes, metric,
    #                         initialState, False)
    how_many = int(round(nbEpisodes / nEpisodesPerJob))
    out = Parallel(
        n_jobs=n_jobs, verbose=2,
    )(
        delayed(_eval_and_render)(gym.make(mdp.spec.id), policy, nEpisodesPerJob, metric, initialState)
        for _ in range(how_many))

    # out is a list of quadruplet: mean J, 95% conf lev J, mean steps, 95% conf lev steps
    # (confidence level should be 0 or NaN)
    V = np.array(out)
    return V[:, 0].mean(), 2 * V[:, 0].std() / np.sqrt(nbEpisodes), \
           V[:, 1].mean(), 2 * V[:, 1].std() / np.sqrt(nbEpisodes)


def evaluate_policy(mdp, policy, nbEpisodes=1,
                    metric='discounted', initialState=None, render=False,
                    n_jobs=-1, nEpisodesPerJob=10):
    """
    This function evaluate a policy on the given environment w.r.t.
    the specified metric by executing multiple episode.
    Params:
        policy (object): a policy object (method drawAction is expected)
        nbEpisodes (int): the number of episodes to execute
        metric (string): the evaluation metric ['discounted', 'average']
        initialState (np.array, None): initial state where to start the episode.
                                If None the initial state is selected by the mdp.
        render (bool): flag indicating whether to visualize the behavior of
                        the policy
    Return:
        metric (float): the selected evaluation metric
        confidence (float): 95% confidence level for the provided metric
    """
    assert metric in ['discounted', 'average'], "unsupported metric for evaluation"
    if render:
        return _eval_and_render(mdp, policy, nbEpisodes, metric, initialState, True)
    else:
        return _parallel_eval(mdp, policy, nbEpisodes, metric, initialState, n_jobs, nEpisodesPerJob)


def collectEpisodes(mdp, policy=None, nbEpisodes=1, n_jobs=1):
    out = Parallel(
        n_jobs=n_jobs, verbose=2,
    )(
        delayed(collectEpisode)(gym.make(mdp.spec.id), policy)
        for i in range(nbEpisodes))

    # out is a list of np.array, each one representing an episode
    # merge the results
    data = np.concatenate(out, axis=0)
    return data


def collectEpisode(mdp, policy=None):
    """
    This function can be used to collect a dataset running an episode
    from the environment using a given policy.

    Params:
        policy (object): an object that can be evaluated in order to get
                         an action

    Returns:
        - a dataset composed of:
            - a flag indicating the end of an episode
            - state
            - action
            - reward
            - next state
            - a flag indicating whether the reached state is absorbing
    """
    done = False
    t = 0
    H = None
    data = list()
    action = None
    if hasattr(mdp, 'horizon'):
        H = mdp.horizon
    state = mdp.reset()
    stateDim, actionDim = getSpaceInfo(mdp)
    assert len(state.shape) == 1
    while (t < H) and (not done):
        if policy:
            action = policy.drawAction(state)
        else:
            action = mdp.action_space.sample()
        nextState, reward, done, _ = mdp.step(action)

        # TODO: should look the dimension of the action
        action = np.reshape(action, (actionDim))

        if not done:
            if t < mdp.horizon:
                # newEl = np.column_stack((0, state, action, reward, nextState, 0)).ravel()
                newEl = [0] + state.tolist() + action.tolist() + [reward] + \
                        nextState.tolist() + [0]
            else:
                # newEl = np.column_stack((1, state, action, reward, nextState, 0)).ravel()
                newEl = [1] + state.tolist() + action.tolist() + [reward] + \
                        nextState.tolist() + [0]
        else:
            # newEl = np.column_stack((1, state, action, reward, nextState, 1)).ravel()
            newEl = [1] + state.tolist() + action.tolist() + \
                    [reward] + nextState.tolist() + [1]

        # assert len(newEl.shape) == 1
        # data.append(newEl.tolist())
        data.append(newEl)
        state = nextState[:]
        t += 1

    return np.array(data)
