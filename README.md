# Goal Conditioned Trajectory Generation With Autoregressive Transformer

**Note**

- This code based on the [official code](https://github.com/eloialonso/iris) of [Transformers are Sample-Efficient World Models](https://openreview.net/forum?id=vhFu1Acb0xb)


## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with python 3.10.11, torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
  - Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.
- To access the pretrained model, you'll need to have [Git LFS](https://git-lfs.github.com/) installed.


## Training

For the training and evaluating the goal-conditioned world model, please follow the steps below.
#### 1. Get the pre-trained world model weights from IRIS
First, clone the pretrained model repository of IRIS [here](https://github.com/eloialonso/iris_pretrained_models.git). Ensure you have Git LFS installed in advance.

#### 2. Move the weights file to `checkpoints/last.pt`
Please choose the model weights according to your training environment. Move the weights file to `checkpoints/` directory, and rename it to `last.pt`. For this work, `Breakout.pt` was used.

#### 3. Configure settings
- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- In `config/trainer.yaml`, please check `initialization.path_to_checkpoint`. Replace `/path/to/goal-conditioned-iris/checkpoints/last.pt` appropriately to your source code path.
- Also, check `common.planning_steps` and set this as you want. This denotes the number of states between the start and goal state including both states. (default: `5`)
- If you are trying to train on the environment other than Breakout, please check out `config/env/default.yaml` and change `train.id` to the environment you want. (default: `BreakoutNoFrameskip-v4`)

#### 4. Run Notebooks
After the configuration, you can run the following notebooks sequentially:
1. `src/0_Expand_Action_Space.ipynb`
2. `src/1_Train_World_Models.ipynb`
3. `src/2_Evaluate_World_Models.ipynb`

Each notebook contains full instructions and explanations. Simply adhering to them should be sufficient.

#### Note: Generating Custom Trajectories (For OOD Evaluation)
Please run `./scripts/play.sh -k` on bash, and play until the game is over.
The trajectories are recorded whenever the game resets, and saved as `custom_trajectories/[date_and_time_you'd_played].pt`.

Currently, `custom_trajectories/sample.pt` are stored as a sample trajectory that was used for OOD Evaluation in this work.

## Pre-trained models
Pre-trained goal-conditioned model weights are available [here](https://drive.google.com/drive/folders/1n33Fyyu-OdD9K-wBzk2xMxloT_HGaSbw?usp=sharing).

- There are three models trained with different `planning_steps` (`t=2,5,10`). Each weights file is named as `t[x]_50k_last.pt` where `[x]`  refers to the `planning_steps` parameter. Each model is trained for 50k gradient descent steps.
- With these model weights, you can skip the training and directly evaluate the pretrained models. Please move the weights file to `/path/to/goal-conditioned-iris/src/outputs/checkpoints/epoch_250/last.pt` and run `src/2_Evaluate_World_Models.ipynb` to evaluate.

## Slides
Please check out this [link for presentation](https://docs.google.com/presentation/d/1bMhNEsRodrUBCJfCfROuLk_t7hw4WBSq/edit?usp=drive_link&ouid=117473038262541831898&rtpof=true&sd=true).
