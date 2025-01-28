# ICLR 2025: Model Compression and Training Efficiency Paper List

This repository serves as a curated collection of the latest research papers accepted at ICLR 2025 in the field of **Model Compression** and **Training Efficiency**. The goal is to provide researchers and practitioners with an organized and up-to-date resource for understanding advancements in these critical areas.

## 📌 Purpose
- Track and showcase groundbreaking research in **model compression** techniques such as pruning, quantization, distillation, and efficient architectures.
- Highlight advancements in **training efficiency** including optimization, resource-efficient methods, and training algorithms designed for scalability.

## 🗂 Paper List
Below is the list of accepted papers in **Model Compression** and **Training Efficiency**. Papers are organized alphabetically by title.

Forget the Data and Fine-Tuning! Just Fold the Network to Compress
- https://openreview.net/forum?id=W2Wkp9MQsF
- Data-free and fine-tuning-free structured compression

Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models
- https://openreview.net/forum?id=3BhZCfJ73Y
- Dynamic pruning, according to prompt

You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning
- https://openreview.net/forum?id=5RZoYIT3u6
- reformulating model pruning as a policy learning process

Optimal Brain Apoptosis
- https://openreview.net/forum?id=88rjm6AXoC
-  Optimal Brain Apoptosis (OBA), a novel pruning method that calculates the Hessian-vector product value directly for each parameter. By decomposing the Hessian matrix across network layers and identifying conditions under which inter-layer Hessian submatrices are non-zero, we propose a highly efficient technique for computing the second-order Taylor expansion of parameters.

LLaMaFlex: Many-in-one LLMs via Generalized Pruning and Weight Sharing
- https://openreview.net/forum?id=AyC4uxx2HW
- Training + Pruning

OATS: Outlier-Aware Pruning Through Sparse and Low Rank Decomposition
- https://openreview.net/forum?id=DLDuVbxORA
-  that compresses the model weights by approximating each weight matrix as the sum of a sparse matrix and a low-rank matrix

Two Sparse Matrices are Better than One: Sparsifying Neural Networks with Double Sparse Factorization
- https://openreview.net/forum?id=DwiwOcK1B7
- we present Double Sparse Factorization (DSF), where we factorize each weight matrix into two sparse matrices. Although solving this problem exactly is computationally infeasible, we propose an efficient heuristic based on alternating minimization via ADMM that achieves state-of-the-art results, enabling unprecedented sparsification of neural networks. 


Streamlining Redundant Layers to Compress Large Language Models
- https://openreview.net/forum?id=IC5RJvRoMp
- If the input and output hidden states of a particular layer are highly similar, such as exhibiting high cosine similarity, we can say that the layer has a small impact on adjusting the hidden states


QP-SNN: Quantized and Pruned Spiking Neural Networks
- https://openreview.net/forum?id=MiPyle6Jef


Adaptive Pruning of Pretrained Transformer via Differential Inclusions
- https://openreview.net/forum?id=WA84oMWHaH

Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing
- https://openreview.net/forum?id=WOt1owGfuN
- three main stages: probing, history-informed pruning, and full inference.


Revisiting Delta-Parameter Pruning For Fine-Tuned Models
- https://openreview.net/forum?id=avSocG0oFA

Find A Winning Sign: Sign Is All We Need to Win the Lottery
- https://openreview.net/forum?id=cLtE4qoPlD
- a signed mask, a binary mask with parameter sign information, can transfer the capability to achieve strong generalization after training (i.e., generalization potential) to a randomly initialized network

Preserving Deep Representations in One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework
- OpenReview Link: https://openreview.net/forum?id=eNQp79A5Oz
- Keywords: Neural Network Pruning;Structured Pruning;Optimization;Hessian-free Optimization

Probabilistic Neural Pruning via Sparsity Evolutionary Fokker-Planck-Kolmogorov Equation
- OpenReview Link: https://openreview.net/forum?id=hJ1BaJ5ELp
- Keywords: Optimization for Deep Network;Probabilistic Method;Machine learning;Model compression

MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More
- OpenReview Link: https://openreview.net/forum?id=hheFYjOsWO
- Keywords: Mixture-of-Expert;LLM;Quantization;Pruning

DPaI: Differentiable Pruning at Initialization with Node-Path Balance Principle
- OpenReview Link: https://openreview.net/forum?id=hvLBTpiDt3
- Keywords: Prunning at Initialization;Sparsity;Neural Architecture Search

ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning
- OpenReview Link: https://openreview.net/forum?id=kSdWcw5mkp
- Keywords: diffusion models;concept editing;pruning
- first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. 

ThinK: Thinner Key Cache by Query-Driven Pruning
- OpenReview Link: https://openreview.net/forum?id=n0OtGl6VGb
- Keywords: Large Language Models; KV Cache Compression; KV Cache Pruning

The Unreasonable Ineffectiveness of the Deeper Layers
- OpenReview Link: https://openreview.net/forum?id=ngmEcEer8a
- Keywords: NLP;Pruning;Science of Deep Learning;Efficient Inference

Beyond Linear Approximations: A Novel Pruning Approach for Attention Matrix
- OpenReview Link: https://openreview.net/forum?id=sgbI8Pxwie
- Keywords: Weights Pruning;Attention Approximation;Gradient Descent Optimization

Rethinking Sparse Scaling through the Lens of Average Active Parameter Count
- OpenReview Link: https://openreview.net/forum?id=ud8FtE1N4N
- Keywords: pruning;sparsity;large language model;pretraining
- Maybe related to unfolding #todo #ModelFolding 


Beware of Calibration Data for Pruning Large Language Models
- OpenReview Link: https://openreview.net/forum?id=x83w6yGIWb
- Keywords: calibration data;post-training pruning;large language models


Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization
- OpenReview Link: https://openreview.net/forum?id=gx1wHnf5Vp
- Keywords: mixture of experts;large language models;continual pre-training #sparsetraining 


SLoPe: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs
- OpenReview Link: https://openreview.net/forum?id=lqHv6dxBkj
- Keywords: sparse training;low rank adapter;LLM;optimization #sparsetraining


MoLEx: Mixture of Layer Experts for Fine-tuning with Sparse Upcycling
- OpenReview Link: https://openreview.net/forum?id=rWui9vLhOc
- Keywords: Parameter efficient fine-tuning;mixture of experts;sparse upcycling


> **Note:** This list will be continuously updated as new papers are reviewed and added.


## 💡 Why Focus on Model Compression and Training Efficiency?
As AI models grow larger and more resource-intensive, the need for efficient training and deployment becomes critical. These fields address challenges such as:
- Reducing computational and memory requirements.
- Enabling deployment on edge devices and resource-constrained environments.
- Enhancing scalability and sustainability in AI research.

## 🌟 Acknowledgements
This repository is inspired by the community of researchers and developers who are driving innovation in efficient AI methods. Special thanks to the authors contributing to ICLR 2025.

---

**📬 Stay Updated:** Watch this repository to get notifications for updates.


**✨ License:** [MIT License](LICENSE)
