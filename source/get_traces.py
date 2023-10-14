import numpy as np

from utils import Trace, State


def get_traces(environment, agent, args):
    """Obtain traces and state dictionary"""
    execution_traces, states_dictionary = [], {}
    for i in range(args.n_traces):
        if args.rand_rate == 0:
            get_single_trace(environment, agent, i, execution_traces,
                             states_dictionary, args)
        else:
            get_single_trace_with_random_actions(
                environment, agent, i, execution_traces,
                states_dictionary, args)
        if args.verbose:
            print(f"\tTrace {i} {15 * '-' + '>'} Obtained")
    if args.verbose:
        print(f"Highlights {15 * '-' + '>'} Traces & States Generated")
    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, agent_traces, states_dict, args):
    """Implement a single trace while using the Trace and State classes"""
    trace = Trace()
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


def get_single_trace_with_random_actions(
        env, agent, trace_idx, agent_traces, states_dict, args
):
    """Implement a single trace while using the Trace and State classes.
    random actions are taken
    """
    trace = Trace()
    curr_obs, info = env.reset()
    done = False
    is_random_action = False
    count_rand_actions = 0
    rand_action = None
    while not done:
        curr_obs = np.array(curr_obs).flatten()
        """Generate State"""
        state_img = env.render(mode='rgb_array')
        state_q_values = agent.get_state_action_values(curr_obs)
        features = None  # TODO implement here
        state_id = (trace_idx, trace.length)
        states_dict[state_id] = State(state_id, curr_obs, state_q_values,
                                      features, state_img)

        if not is_random_action:
            is_random_action = np.random.random() < args.rand_rate

        if is_random_action and count_rand_actions < args.n_rand_actions:
            """choose an action at random and repeat it several times"""
            if rand_action is None:
                rand_action = np.random.choice(agent.action_space)
            action = rand_action
            count_rand_actions += 1

        else:
            """take an action based on policy"""
            action = agent.act(curr_obs)

        obs, r, done, trunc, infos = env.step(action)
        # if done then r = r + max(state_q_values)
        obs = np.array(obs).flatten()

        """Add step and state to trace"""
        trace.update(curr_obs, obs, r, done, infos, action, state_id,
                     is_random_action)

        """Update observation to the following one"""
        curr_obs = obs

        if count_rand_actions == args.n_rand_actions:
            is_random_action = False
            count_rand_actions = 0
            rand_action = None

    """Generate Final State and add to States.
    This state is the one seen once the episode terminates.
    """
    state_img = env.render(mode='rgb_array')
    state_q_values = agent.get_state_action_values(curr_obs)
    features = None  # TODO implement here
    state_id = (trace_idx, trace.length)
    states_dict[state_id] = State(state_id, curr_obs, state_q_values,
                                  features, state_img)

    agent_traces.append(trace)
