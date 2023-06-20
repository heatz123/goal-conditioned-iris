# Goal Conditioned Trajectory Generation With Transformer World Models

**Note**

- This work is based on the [official code](https://github.com/eloialonso/iris) of [Transformers are Sample-Efficient World Models](https://openreview.net/forum?id=vhFu1Acb0xb)


## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with python=3.10.11, torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
  - Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.
- [Git LFS](https://git-lfs.github.com/) should be installed to get the pretrained model.


## Training

For training and evaluation of the goal-conditioned world model, please follow the instructions below.
#### 1. Get the pre-trained world model weights from IRIS
First, clone the pretrained model repository of IRIS [here](https://github.com/eloialonso/iris_pretrained_models.git). You should have `Git LFS` installed in advance.

#### 2. Move the weights file to `checkpoints/last.pt`
Please choose the model weights according to your training environment to `checkpoints/`, and rename it to `last.pt`. For this work, `Breakout.pt` was used.

#### 3. Configure settings
- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- In `config/trainer.yaml`, please check `initialization.path_to_checkpoint`. Replace `/path/to/goal-conditioned-iris/checkpoints/last.pt` appropriately to your source code path.
- Also check `common.planning_steps` and set this as you want. This denotes the number of states between the start and goal state including both states. (default: `5`)
- If you are trying to train on the environment other than Breakout, please check the `config/env/default.yaml` and change `train.id` to the environment you want. (default: `BreakoutNoFrameskip-v4`)

#### 4. Run Notebooks
Now you have to sequentially run the following notebooks:
1. `src/0_Expand_Action_Space.ipynb`
2. `src/1_Train_World_Models.ipynb`
3. `src/2_Evaluate_World_Models.ipynb`

Each notebook contains full instructions and explanations. Simply adhering to them should be sufficient.

#### Note: Generating Custom Trajectories (For OOD Evaluation)
Please run `./scripts/play.sh -k` on bash, and play until the game is over.
The trajectories are recorded whenever the reset game is called, and saved as `custom_trajectories/[date_and_time_you'd_played].pt`.

## Pretrained models

Pretrained goal-conditioned model weights are available [here](https://drive.google.com/drive/folders/1n33Fyyu-OdD9K-wBzk2xMxloT_HGaSbw?usp=sharing).

- There are three models trained with different planning_steps (`t=2,5,9`). Each weights file is named as `t[x]_50k_last.pt`, and the `[x]` here refers to the `planning_steps` parameter. Each model is trained for 50k gradient descent steps.
- You can skip the training and directly evaluate the pretrained models. Please move the weights file to `/path/to/goal-conditioned-iris/src/outputs/checkpoints/epoch_250/last.pt` and run `src/2_Evaluate_World_Models.ipynb`.
