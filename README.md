# Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion

<p align="center">
  <a href="https://arxiv.org/abs/2408.00280">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="SNN Temporal Fusion Paper on arXiv">
  </a>
</p>


This work presents a novel *temporal fusion* method, specifically designed to expedite the propagation dynamics of SNNs on GPU platforms, which serves as an enhancement to the current significant approaches for handling deep learning tasks with SNNs. This method underwent thorough validation through extensive experiments in both authentic training scenarios and idealized conditions, confirming its efficacy and adaptability for single and multi-GPU systems. 

## Experiment Setup and Execution

All experiments in this work were conducted under the configuration of `Ubuntu 22.04`, `CUDA 12.4`, and `PyTorch 2.3.1`.

### Single GPU

To start the experiments, run the following code:

```shell
cd ./single_gpu/
python single_gpu_test.py --device 0 --dataset MNIST --arch Spiking-ResNet18
```
In single GPU experiments, the dataset can be selected from `MNIST`, `CIFAR-10`, `N-MNIST`, and `DvsGesture`. The network architecture can be specified using the `--arch` parameter, with options including `Spiking-ResNet18`, `Spiking-ResNet34`, and `Spiking-ResNet50`. If the dataset `N-MNIST` or `DvsGesture` is chosen, please ensure that you have installed the configuration of `SpikingJelly 0.0.0.0.14` beforehand.

### Multi-GPU

To start the experiments, run the following code:

```shell
cd ./multi_gpu/
chmod 755 ./MultiGPUTest
./MultiGPUTest
```

The number of GPUs and the number of time steps for the model need to be specified at runtime. Please ensure that the environment has a sufficient number of GPUs available.

## Citing This Work

```latex
@InProceedings{snn_temporal_fusion_2024,
    title={Towards scalable {GPU}-accelerated {SNN} training via temporal fusion}, 
    author={Yanchen Li and Jiachun Li and Kebin Sun and Luziwei Leng and Ran Cheng},
    year={2024},
    booktitle={Artificial Neural Networks and Machine Learning -- ICANN 2024},
    publisher={Springer Nature Switzerland},
    address={Cham},
    pages={58--73},
    isbn={978-3-031-72341-4},
    doi={10.1007/978-3-031-72341-4_5},
}
```

