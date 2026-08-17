# Pytorch code from I-HER thesis experiments

Built upon this [DDPG+HER pytorch implementation](https://github.com/TianhongDai/hindsight-experience-replay).

## Structure and files

- `dynamics/`: contains class defining the feed-forward dynamics model
- `her_modules/`: code for hindsight experience replay goal and reward resampling
- `mpi_utils/`: functions for parallelizing across multiple threads. Also contains observations normalization functions.
- `rl_modules/`: main RL code
    - `models.py`contains actor critic networks
    - `replay_buffer.py`contains replay buffer
    - The `ddpg_agent___.py` files contain the main training loop code but are a bit of a mess. I seem to have created a new file for each major iteration. Based on the order of lines 10-18 in `train.py`, it went `_simple_dynamics` -> `_iter_dynamics`-> ... -> `_differ_real_imag2` -> `_differ_real_delta`.
    - So, `ddpg_differ_real_delta` is the full final method from the thesis (i.e., the one you should use).
    - `ddpg_agent.py` should just run standard HER.
    - `ddpg_real_transfer` is related to transfering a dynamics learned in FetchPick&Place to FetchPush.
- `worldmodels/`: contains code used to create 'imaginary' environments from the learned dynamics models.     
- `arguments.py`:
    - The hyperparameters that can be set (see examples in 'Training' section below).
    - Be careful here - I think there are some hyperparameters/things I had to change manually when running different experiments. The only one I can remember is line 57 of `ddpg_differ_real_delta` - here you should change from `FetchPushEnsembleDelta` to `FetchReachEnsembleDelta` if running on the FetchReach environment (`FetchPushEnsembleDelta` works for Push, Pick, and Slide though)
- `demo_imag_delta.py`can be used for testing saved models and visualising results. (`demo.py` is the original DDPG+HER demo script)


## Training

To run training:

    mpirun -np 8 python3 -u train.py --env-name='FetchPush-v1' --n-epochs=100 --seed=123 2>&1 | tee train_results.log

`-np` specifies the number of MPI processes that will be run in parallel. Expect inferior performance if less than 8 are used. See `commands.txt`for more example commands. See the main hyperparameters flags in the command used in the `her.sh` file.

You can change the `train.py` script to choose which 'ddpg_trainer' to use (`ddpg_differ_real_delta` is recommended).


### Sonic

To run on sonic:

    sbatch her.sh

Modify the `her.sh` script and hyperparameters as needed. The current script restricts the job to nodes 66 or 67 (these were the only ones with Mujoco installed), but this may no longer be necessary.
