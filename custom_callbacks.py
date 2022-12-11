from util import plot_trajs
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class TrajLoggingCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.traj = []
        self.trajs = []

    def _on_step(self) -> bool:
        obs = self.model.env.get_attr("observation")[0]
        done = self.locals["dones"][0]
        self.traj.append(obs["position"])
        if done:
            self.trajs.append(self.traj[:-1])
            self.traj = []
        return True


class EvalTrajLoggingCallBack:
    def __init__(self, env):
        self.traj = []
        self.env = env
        self.cnt = 0

    def __call__(self, locals, _):
        obs = locals["observations"]["position"]
        visited = locals["observations"]["visited"][-1]
        n_eval = locals["n_eval_episodes"]
        done = locals["dones"][0]
        self.traj.append(obs)
        if done:
            plot_trajs(
                self.env,
                [self.traj[:-1]],
                visited,
                "result/trajs_eval_{}.png".format(self.cnt % n_eval),
                alpha=1,
            )
            self.cnt += 1
            self.traj = []


class CustomEvalCallBack(EvalCallback):
    """Callback for evaluation"""

    def init_attrs(self, eval_env):
        self.reward_list = []
        self.traj = EvalTrajLoggingCallBack(eval_env)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, callback=self.traj
            )
            self.reward_list.append(mean_reward)
            print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return True
