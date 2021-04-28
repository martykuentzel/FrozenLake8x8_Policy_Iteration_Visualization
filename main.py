#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation
from matplotlib import colors
import random
import gym
from matplotlib.colors import ListedColormap
from gym import wrappers

#uncomment when run in interactive mode inside vscode
#%matplotlib widget

###############
# Polciy Iteration
###############
def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def simulate_policy(env, policy, gamma = 1.0, n = 1000):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def policy_improvement(v, gamma = 1.0):
    """ Improves the policy given a value-function """
    policy = np.zeros(env.nS)
    max_q = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_next]) for p, s_next, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
        max_q[s] = np.max(q_sa)
    return policy, np.sum(max_q)


def policy_evaluation(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy. """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_next]) for p, s_next, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS)) # initialize a random policy
    max_iterations = 200000

    for i in range(max_iterations):
        # Bellman Expectation Equation
        v = policy_evaluation(env, policy, gamma)
        old_policy = np.copy(policy)
        # Greedy Policy Improvement
        policy, max_q = policy_improvement(v, gamma)

        # Check for convergence
        if (np.all(policy == old_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            print("max_q:", max_q)
            print("v:", np.sum(v))
            print("Note: V(s) for Policy Pi equals max q(s,a) => the Bellman optimality equation is satisfied.")
            break
    return policy, v

###############
# Create Plot
###############
def create_plot(ax_v,ax_as, ax_q, ax_final):
    cmap = colors.ListedColormap(['yellow','lightblue','gray','green'])
    norm = colors.BoundaryNorm(np.arange(-0.5,4), cmap.N)
    data = np.zeros((DIM_GAME,DIM_GAME))

    im_v = ax_v.imshow(data, vmin=0, vmax=1.5, cmap="Greens", aspect='auto')
    v_texts = []
    for i in range(data.shape[0]):
        row = []
        for j in range(data.shape[1]):
            row.append(ax_v.text(j,i, "", va="center", ha="center"))
        v_texts.append(row) 

    im_final = ax_final.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    im_as = ax_as.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    c = np.zeros_like(data)
    as_texts = []
    final_texts = []
    for i in range(data.shape[0]):
        row = []
        row_final = []
        for j in range(data.shape[1]):
            c[i,j] = field[desc[i][j]]
            row.append(ax_as.text(j,i, "", va="center", ha="center"))
            row_final.append(ax_final.text(j,i, "", va="center", ha="center"))
        as_texts.append(row) 
        final_texts.append(row_final) 
    im_as.set_array(c)
    im_final.set_array(c)

    for ax in [ax_v, ax_as, ax_final]:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.yaxis.set_major_locator(mticker.IndexLocator(1,0))
        ax.xaxis.set_major_locator(mticker.IndexLocator(1,0))
        ax.grid(color="k")

    ax_q.set_xlabel("Number of Iterations")
    ax_q.set_ylabel("Total Q of all states")
    line, = ax_q.plot(0, 0)
    ax_q.set_xlim([0,14])
    ax_q.set_ylim([0,45])
    ax_v.title.set_text('V')
    ax_as.title.set_text('Actions for current Policy')
    tags = []
    tags.append(ax_q.text(6, -10, f"Total Q:", ha="center", fontsize=12, bbox={"facecolor":"green", "alpha":0.7, "pad":5}))
    for i in range(1,4):
        tags.append(ax_final.text(1.5, 8+(i/2), "", fontsize=12))

    return v_texts,as_texts, final_texts, line, tags, im_v

###############
# Animation
###############
def update_p_v_plot(v, policy):
    v_2d = np.round(v.reshape(DIM_GAME,DIM_GAME),2)
    p_2d = policy.reshape(DIM_GAME,DIM_GAME)
    im_v.set_array(v_2d)
    for i in range(DIM_GAME):
         for j in range(DIM_GAME):
             v_texts[i][j].set_text(v_2d[i,j])
             as_texts[i][j].set_text(arrows[p_2d[i,j]])

def update_q_plot(i, q):
    q_arr.append(q)
    line.set_data(np.arange(i+1), q_arr)
    tags[0].set_text(f"Total Q: {round(q,2)}")

def init():
    for row in v_texts:
        for text in row:
            text.set_text("")

def animate(i):
    global converged
    global state
    global total_reward
    global step_idx
    if not converged:
        v = policy_evaluation(env, policies[i], gamma)
        update_p_v_plot(v, policies[i])
        policy, q = policy_improvement(v, gamma)
        policies.append(policy)
        if np.all(policies[i-1] == policies[i]):
            converged=True
        update_q_plot(i, q)
    else:
        ax_q.title.set_text('Converged!')
        ax_final.title.set_text('Lets try the Policy!')
        step_idx +=1
        action = int(policies[-1][state])

        s_x, s_y = int(state/8), state%8
        final_texts[s_x][s_y].set_text("")

        state, _, done , _ = env.step(action)

        s_x, s_y = int(state/8), state%8
        final_texts[s_x][s_y].set_text(f"0\n{arrows[action]}")

        tags[1].set_text(f"state: ({s_x},{s_y})")
        tags[2].set_text(f"Action: {direction[action]}")
        tags[3].set_text(f"Step: {step_idx}")
        if done:
            final_texts[s_x][s_y].set_text("")
            state=env.reset()
            step_idx=0

###############
# Setup
###############

env_name  = 'FrozenLake8x8-v0'
env = gym.make(env_name)
DIM_GAME=8
VISUALIZATION_ENABLED=True

# Actions:
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
arrows = {0:"<", 1:"v", 2:">", 3:"^"}
direction = {0:"Left", 1:"Down", 2:"Right", 3:"Up"}
desc = [[c.decode('utf-8') for c in line] for line in env.desc]

# used for coloring the grid accoridng to start, frozen, hole and goal cell
field = {'S':-0.5, 'F':1, 'H':2, 'G':4}

# initilize all variables that will be used inside the animate function
# it basically runs an animated version of the policy_iteration function
policy = np.random.choice(env.nA, size=(env.nS)) 
policies = [policy]
gamma = 1.0
q_arr = []
converged=False
total_reward=0
state=env.reset()
gamma = 1.0
step_idx=0

if VISUALIZATION_ENABLED:
    fig, ((ax_v, ax_as), (ax_q, ax_final)) = plt.subplots(2,2,figsize=(15,8))
    fig.suptitle('Policy Iteration Visualization', fontsize=12)

    # very hacky setup of the plot :)
    v_texts, as_texts,final_texts, line, tags, im_v = create_plot(ax_v,ax_as, ax_q, ax_final)

    ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, #blit=True,
                                            frames=np.arange(10000), interval=800)
    #ani.save("policy_iteration.gif", fps=1,writer="imagemagick")
    plt.show()
else:
    optimal_policy, v = policy_iteration(env, gamma = 1.0)
    scores = simulate_policy(env, optimal_policy, gamma = 1.0)
    print('Average scores (Aggregated Rewards for finishing the Game) = ', np.mean(scores))
