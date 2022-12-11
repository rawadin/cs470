from util import *
from custom_callbacks import *
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import *
from gym.wrappers import *
import uuid
import matplotlib.pyplot as plt
import numpy as np

register_env()

env_id = "Gridworld-v1"
SEED = 101
set_random_seed(SEED)

env = gym.make(env_id)
env.seed(SEED)
# env = Monitor(env, "/tmp/env-" + str(uuid.uuid4()))
eval_env = gym.make(env_id)
eval_env.seed(SEED)
# eval_env = Monitor(eval_env, "/tmp/eval-env-" + str(uuid.uuid4()))

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logs/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5, min_evals=5, verbose=1
)
custom_eval_callback = CustomEvalCallBack(
    eval_env=eval_env, eval_freq=1000, callback_after_eval=stop_train_callback
)
custom_eval_callback.init_attrs(eval_env)
loggin_callback = TrajLoggingCallBack()

model = DQN(
    policy="MultiInputPolicy",
    env=env,
    seed=SEED,
    learning_starts=0,  # Decide warming up steps
    batch_size=64,  # Batch size for the neural network
    learning_rate=3e-4,  # Learning rate for the neural network
    buffer_size=30000,  # The size of stored transitions from the past
    exploration_initial_eps=1.0,  # Exploration rate will be
    exploration_fraction=0.2,  # gradually descreasing from exploration_inital_eps to exploration_final_eps
    exploration_final_eps=0.1,  # for exploration_fraction amount of times
    target_update_interval=250,  # Update interval of the target neural network
    train_freq=4,  # How often the neural netwok to be updated (once per N steps)
    gradient_steps=-1,  # Number of updates per batch  (if -1, set to batch_size)
    policy_kwargs=dict(net_arch=[64, 64]),
    # The shape of the neural network
    gamma=0.9,  # Discount Factor
)

model.learn(
    total_timesteps=100000,
    callback=[
        custom_eval_callback,
        loggin_callback,
        checkpoint_callback,
    ],
    # progress_bar=True,
)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(custom_eval_callback.reward_list, color="blue")
ax.set_xlabel("Number of timesteps ($\\times 10^3$)")
ax.set_ylabel("Sum of rewards during episode")
plt.savefig("result/result.png")
plt.clf()

plot_trajs(env, loggin_callback.trajs, np.zeros(2500), "result/trajs.png")

print("end")
