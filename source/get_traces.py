import numpy as np

from utils import Trace, State


def get_traces(environment, agent, args):
    """Obtain traces and state dictionary"""
    execution_traces, states_dictionary = [], {}
    for i in range(args.n_traces):
        get_single_trace(environment, agent, i, execution_traces,
                         states_dictionary, args)
        if args.verbose:
            print(f"\tTrace {i} {15*'-'+'>'} Obtained")
    if args.verbose:
        print(f"Highlights {15*'-'+'>'} Traces & States Generated")
    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, agent_traces, states_dict, args):
    """Implement a single trace while using the Trace and State classes"""
    trace = Trace()
    # ********* Implement here *****************
    curr_obs, info = env.reset()
    done = False
    while not done:
        curr_obs = np.array(curr_obs).flatten()

        """Generate State"""
        state_img = env.render(mode='rgb_array')
        state_q_values = agent.get_state_action_values(curr_obs)
        features = None  # TODO implement here
        state_id = (trace_idx, trace.length)
        states_dict[state_id] = State(state_id, curr_obs, state_q_values,
                                      features, state_img)

        """take an action and update environment"""
        a = agent.act(curr_obs)
        obs, r, done, trunc, infos = env.step(a)
        # if done then r = r + max(state_q_values)
        obs = np.array(obs).flatten()

        """Add step and state to trace"""
        trace.update(curr_obs, obs, r, done, infos, a, state_id)

        """Update observation to the following one"""
        curr_obs = obs

    """Generate Final State and add to States.
    This state the one seen once the episode terminates.
    """
    state_img = env.render(mode='rgb_array')
    state_q_values = agent.get_state_action_values(curr_obs)
    features = None  # TODO implement here
    state_id = (trace_idx, trace.length)
    states_dict[state_id] = State(state_id, curr_obs, state_q_values,
                                  features, state_img)

    agent_traces.append(trace)
