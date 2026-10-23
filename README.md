# The implementation of *Clockwork Low-Rank Adaptation with Calibrated Classifiers for Online Continual Learning*.


## 🚀 Abstract

As a continual  learning paradigm, online continual learning (OCL) receives  evanescent data streams and requires models to  evolve to fit non-stationary data automatically. That leads to three challenges: confusion boundary over training trajectory, insufficient knowledge accumulation, and severe stability-plasticity conundrum. Inspired from the Dual-Process Theory on human cognition, OCL requires a clock-like mechanism that can adaptively regulate model updates across multiple timescales. Hence, we incorporate clockwork systems with Low-rank adaptation (LoRA) together, and propose the Clockwork Low-Rank Adaptation Network with Calibrated Classifiers (CLoRANet). The core component is the CLoRA module, which formulates "clock signals" with two gradient statistics: the gradient energy reflects update intensity and the gradient conflict measures interference between new and old knowledge. Upon these clock signals, the CLoRA module defines event-triggered ticks, in order to monitor abrupt changes and gradual changes in gradient statistics and determine when and how the model updates. Combining it with the Synthesis Classifier and Semantic Drift Calibration modules, CLoRANet maintains calibrated and robust feature distributions over non-stationary data streams.

## 🖥️ Environment

- Python 3.11
- CUDA 20.2

``` bash
conda create -n clora python=3.11
conda activate clora
pip install -r requirements.txt
```

## 📊 Datasets

**NOTE**: Datasets are automatically downloaded in `data/`.  
- This can be changed by changing the `base_path` function in `utils/conf.py` or using the `--base_path` argument.  
- The `data/` folder should not be tracked by _git_ and is created automatically if missing.

Following the rigorous evaluation protocol of Online Continual Learning (OCL), we evaluate on two widely adopted benchmark settings: **Sequential** and **Blurry**.

### Sequential Benchmarks

Sequential benchmarks consist of a stream of distinct tasks with clear boundaries, where each task contains a disjoint set of classes introduced in a temporal order. Although task partitions are defined during dataset construction, the model remains agnostic to task identity and boundary information during both training and evaluation.

Following prior works on pretrained continual learning, we adopt four representative datasets with diverse semantic granularity and scale:

- **(1) Seq-CIFAR100**: A medium-scale benchmark consisting of 100 classes, each with 500 training and 100 test images. The dataset is divided into 10 disjoint tasks, with each task containing 10 classes.
- **(2) Seq-ImageNet-R**: A cross-domain dataset derived from 200 ImageNet classes, characterized by substantial distribution shifts across visual styles, including cartoons, sketches, embroidery, and origami. It contains approximately 24k training images and 6k test images, split into 10 tasks.
- **(3) Seq-CUB200**: A fine-grained classification dataset consisting of 200 bird species, featuring high inter-class similarity and large intra-class variation. The dataset is divided into 10 disjoint tasks, with each task containing 20 classes.
- **(4) Seq-ImageNet-S**: A large-scale dataset with 1000 classes, offering richer semantic diversity and a larger label space. It is partitioned into 20 sequential tasks.

```bash
# Seq-CIFAR100

python main.py --model clora --dataset seq-cifar100-224 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 100 --alpha 10 --beta 3 --gama 1  

# Seq-ImageNet-R
python main.py --model clora --dataset seq-imagenet-r --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 200 --alpha 30 --beta 1 --gama 1  

# Seq-CUB200
python main.py --model clora --dataset seq-cub200 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 200 --alpha 10 --beta 5 --gama 1  

# Seq-ImageNet-S
python main.py --model clora --dataset seq-imagenet-s --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 1000 --alpha 20 --beta 3 --gama 1  
```
### Blurry Benchmarks

The Blurry setting relaxes the assumption of clear task boundaries by allowing classes to reappear across multiple stages, where samples from different classes are interleaved over time. This better reflects realistic scenarios with ambiguous task transitions.

Specifically, we adopt CIFAR100-BlurryK and ImageNet-R-BlurryK, where a proportion K% of samples from other tasks is randomly injected into the current data stream. We consider two mixing ratios K = 10, 30 to evaluate model robustness under different levels of class overlap. Each dataset is divided into 10 disjoint tasks.

```bash
# CIFAR100-Blurry10

python main.py --model clora --dataset seq-cifar100-224-blurry10 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 100  --alpha 10 --beta 3 --gama 1  

# CIFAR100-Blurry30
python main.py --model clora --dataset seq-cifar100-224-blurry30 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 100  --alpha 10 --beta 3 --gama 1  

# ImageNet-R-Blurry10
python main.py --model clora --dataset seq-imagenet-r-blurry10 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 200  --alpha 30 --beta 3 --gama 1  

# ImageNet-R-Blurry30
python main.py --model clora --dataset seq-imagenet-r-blurry30 --lr 0.005 --batch_size 64 --n_epochs 1 --loss_window_length 5 --n_classes 200  --alpha 20 --beta 3 --gama 1  
```

## 🧠 Setup
- 🚀 Use `main.py` or `./utils/main.py` to run experiments.
- 🧩 New models can be added to the `models/` folder.
- 📊 New datasets can be added to the `datasets/` folder.
