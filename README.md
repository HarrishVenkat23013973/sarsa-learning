# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
The Frozen Lake problem is a reinforcement learning task in which an agent must navigate a 4x4 grid (16 states) to reach the goal state. The environment is slippery, meaning the agent has a chance of moving in the opposite direction of the intended action, making it harder to navigate. The agent's goal is to learn an optimal policy to reach the goal while avoiding hazards (like holes in the ice) through trial and error, adjusting its strategy over time.

## SARSA LEARNING ALGORITHM


Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.

For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.

Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.

Update the Q-value for the state-action pair (s,a) using the SARSA update rule.

Update the current state to s′ and the current action to a′.

Repeat steps 3-5 until the episode reaches a terminal state.

After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.

Return the Q-table and the learned policy after completing all episodes.

## SARSA LEARNING FUNCTION
### Name: HARRISHVENKAT V
### Register Number: 212223240049

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    # Write your code here
    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
        next_state, reward, done, _= env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

Q_sarsas, V_sarsas, Q_track_sarsas = [], [], []
for seed in tqdm(SEEDS, desc='All seeds', leave=True):
    random.seed(seed); np.random.seed(seed) ; env.seed(seed)
    Q_sarsa, V_sarsa, pi_sarsa, Q_track_sarsa, pi_track_sarsa = sarsa(env, gamma=gamma, n_episodes=n_episodes)
    Q_sarsas.append(Q_sarsa) ; V_sarsas.append(V_sarsa) ; Q_track_sarsas.append(Q_track_sarsa)
Q_sarsa = np.mean(Q_sarsas, axis=0)
V_sarsa = np.mean(V_sarsas, axis=0)
Q_track_sarsa = np.mean(Q_track_sarsas, axis=0)
del Q_sarsas ; del V_sarsas ; del Q_track_sarsas
```


## OUTPUT:

<img width="1517" height="799" alt="image" src="https://github.com/user-attachments/assets/576766ff-3a5e-4db1-9f21-2a5a63433731" />


#### FVMC

<img width="681" height="852" alt="image" src="https://github.com/user-attachments/assets/09e960e4-e0f4-4b64-a3ba-a92fd59be02b" />

#### SARSA
<img width="895" height="870" alt="image" src="https://github.com/user-attachments/assets/7f3898d6-bb1b-4854-9dbd-9fc55d602ac0" />

#### FVMC 
<img width="1207" height="737" alt="Screenshot 2026-09-07 195235" src="https://github.com/user-attachments/assets/6bf2f8a1-4aca-4baa-8837-40a2f8022c2d" />

#### SARSA

<img width="1760" height="547" alt="Screenshot 2026-09-07 195430" src="https://github.com/user-attachments/assets/906f582d-4f05-4d53-ab54-c2c34e84b4e6" />

<img width="1757" height="606" alt="image" src="https://github.com/user-attachments/assets/a10c5a02-6ca2-46fb-8adb-2a78abab42dd" />

## RESULT:
Thus, SARSA learning successfully trained an agent for optimal policy.
