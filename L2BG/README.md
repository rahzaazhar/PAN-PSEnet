

# Learn to Bind and Grow Neural Structures

This repository is the official implementation of Learn to Bind and Grow Neural Structures. Paper can be found [here](paper.pdf)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
To download data for VDD run
```
bash download_VDD_data.sh
```
copy decathlon_mean_std.pickle to the downloaded folder

## Training

To train the model in the paper, run this command:

```train
python run_expts.py --config_name <name_of_config>
```
Check L2G_config.py for various training configurations and parameters
some parameters in the L2G_config.py are given below
- n_tasks
- datamode: (pmnist/CIFAR100/VDD/mltr)
- lr
- epochs
- alpha
- sim_strat(egy):



