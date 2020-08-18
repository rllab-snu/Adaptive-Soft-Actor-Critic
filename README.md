# Adaptive-Soft-Actor-Critic

Jae In Kim, Mineui Hong, Kyungjae Lee, DongWook Kim, Yong-Lae Park, and Songhwai Oh, "Learning to Walk A Tripod Mobile Robot Using Nonlinear Soft Vibration Actuators with Entropy Adaptive Reinforcement Learning,"  IEEE International Conference on Robotics and Automation (ICRA), May 2020. (RA-L option)

## Installaction
### Prerequisite
```sh
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
### Virtual Environment (Reconmmend)
```sh
virtualenv venv --python=python3.5 (--system-site-packages)
```
If your machine already has tensorflow-gpu package, I reconmmend the option **--system-site-packages** to use tensorflow-gpu.
### Install MuJoCo (Recommend)
```sh
pip install gym[mujoco,robotics]
```
### Install Spinningup with Adaptive Soft Actor Critic
```sh
cd adaptive_soft_actor_critic
pip install -e .
```
### Install Custom Gym
```sh
cd adaptive_soft_actor_critic/custom_gym/
pip install -e .
```
If you want to add a customized environment, see https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
## Reproducing experiments
### Run test
```sh
cd adaptive_soft_actor_critic
python -m spinup.run asac --env HalfCheetah-v2
```

### Run single experiment
```sh
cd adaptive_soft_actor_critic
python -m spinup.run asac --env HalfCheetah-v2 --exp_name half_asac --seed 0 10 20 30 40 50 60 70 80 90
```
Results will be saved in _data_ folder

