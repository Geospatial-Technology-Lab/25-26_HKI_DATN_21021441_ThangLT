"""
Vectorized Environment Wrapper for DRL Training
================================================
Provides parallel environment execution for faster training.
Supports both multiprocessing (SubprocVecEnv) and threading (ThreadVecEnv).
"""

import numpy as np
from multiprocessing import Process, Pipe
from typing import List, Callable, Tuple, Any, Optional
import threading
import queue


class SubprocVecEnv:
    """
    Vectorized environment using subprocesses for parallel rollouts.
    Best for CPU-heavy environments.
    """
    
    def __init__(self, env_fns: List[Callable], start_method: str = 'spawn'):
        """
        Initialize vectorized environment.
        
        Args:
            env_fns: List of functions that create environments
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # Get multiprocessing context
        import multiprocessing as mp
        ctx = mp.get_context(start_method)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.ps = [
            ctx.Process(target=self._worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        for p in self.ps:
            p.daemon = True
            p.start()
        
        # Close worker end of pipes in parent
        for work_remote in self.work_remotes:
            work_remote.close()
        
        # Get observation and action spaces from first env
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions, one per environment
            
        Returns:
            obs: Stacked observations
            rewards: Array of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        
        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)
    
    def step_async(self, actions: np.ndarray):
        """Start async step for all environments"""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for async step to complete"""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    def reset_single(self, env_idx: int) -> np.ndarray:
        """Reset a single environment"""
        self.remotes[env_idx].send(('reset', None))
        return self.remotes[env_idx].recv()
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for p in self.ps:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
        
        self.closed = True
    
    def get_attr(self, attr_name: str) -> List[Any]:
        """Get attribute from all environments"""
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]
    
    @staticmethod
    def _worker(remote, parent_remote, env_fn):
        """Worker process that runs environment"""
        parent_remote.close()
        env = env_fn()
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    obs, reward, done, info = env.step(data)
                    if done:
                        obs = env.reset()
                    remote.send((obs, reward, done, info))
                    
                elif cmd == 'reset':
                    obs = env.reset()
                    remote.send(obs)
                    
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                    
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                    
                elif cmd == 'get_attr':
                    remote.send(getattr(env, data))
                    
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
                    
            except EOFError:
                break


class ThreadVecEnv:
    """
    Vectorized environment using threads for parallel rollouts.
    Faster than SubprocVecEnv for lightweight environments due to lower overhead.
    Best when using GPU, as it avoids pickling issues with CUDA tensors.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Initialize threaded vectorized environment.
        
        Args:
            env_fns: List of functions that create environments
        """
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.closed = False
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments (uses threading for parallel execution)"""
        results = [None] * self.num_envs
        
        def step_env(idx, action):
            obs, reward, done, info = self.envs[idx].step(action)
            if done:
                obs = self.envs[idx].reset()
            results[idx] = (obs, reward, done, info)
        
        threads = [
            threading.Thread(target=step_env, args=(i, actions[i]))
            for i in range(self.num_envs)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)
    
    def reset(self) -> np.ndarray:
        """Reset all environments in parallel"""
        results = [None] * self.num_envs
        
        def reset_env(idx):
            results[idx] = self.envs[idx].reset()
        
        threads = [
            threading.Thread(target=reset_env, args=(i,))
            for i in range(self.num_envs)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        return np.stack(results)
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True


class DummyVecEnv:
    """
    Simple vectorized environment wrapper that runs environments sequentially.
    Useful for debugging and benchmarking.
    """
    
    def __init__(self, env_fns: List[Callable]):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.closed = False
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step environments sequentially"""
        obs_list, reward_list, done_list, info_list = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        return np.stack(obs_list), np.array(reward_list), np.array(done_list), info_list
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        return np.stack([env.reset() for env in self.envs])
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True


def make_vec_env(env_fn: Callable, 
                 num_envs: int = 4, 
                 vec_env_cls: str = 'thread') -> Any:
    """
    Create a vectorized environment.
    
    Args:
        env_fn: Function that creates an environment
        num_envs: Number of parallel environments
        vec_env_cls: Type of vectorized env ('thread', 'subproc', 'dummy')
        
    Returns:
        Vectorized environment
    """
    env_fns = [env_fn for _ in range(num_envs)]
    
    if vec_env_cls == 'thread':
        return ThreadVecEnv(env_fns)
    elif vec_env_cls == 'subproc':
        return SubprocVecEnv(env_fns)
    elif vec_env_cls == 'dummy':
        return DummyVecEnv(env_fns)
    else:
        raise ValueError(f"Unknown vec_env_cls: {vec_env_cls}")


if __name__ == "__main__":
    # Test with a simple environment
    import gym
    
    print("Testing Vectorized Environments")
    print("=" * 50)
    
    # Create test environments
    def make_env():
        return gym.make('CartPole-v1')
    
    # Test ThreadVecEnv
    print("\nTesting ThreadVecEnv...")
    vec_env = make_vec_env(make_env, num_envs=4, vec_env_cls='thread')
    obs = vec_env.reset()
    print(f"  Initial obs shape: {obs.shape}")
    
    for step in range(10):
        actions = np.array([vec_env.action_space.sample() for _ in range(4)])
        obs, rewards, dones, infos = vec_env.step(actions)
    
    print(f"  Final obs shape: {obs.shape}")
    vec_env.close()
    print("  ThreadVecEnv: OK")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
