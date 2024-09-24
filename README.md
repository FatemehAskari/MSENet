# Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms
Implementation of a Few-Shot Image Classification Model based on the Prototypical Network Model and Tested on the MiniImagenet and FC100 Datasets.

For more information, check out our paper on [[arXiv](https://arxiv.org/abs/2409.07989)], [[paperswithcode](https://paperswithcode.com/paper/enhancing-few-shot-image-classification)].

## Model

Our model consists of the following components:
1. We extracted five feature maps from backbone in order to capture both global and task specific features
2. We employ a self-attention mechanism for each feature map obtained from every stage in order to capture more valuable information
3. We incorporate learnable weights at each stage.
4. We propose a novel few-shot classification. We have
significantly improved the accuracy on the MiniImageNet and FC100 datasets.

The final model architecture is as follows:

![Architecture of model](assets/finalmodel.png)

The mapper architecture is as follows:

![Architecture of mapper](assets/attention-module.png)

You can study the model in more detail from this [PDF](finalreport.pdf).

## How to run
For the 5-way 5-shot:
```bash
python train.py --max-epoch 200 --save-epoch 20 --shot 5 --query 10 --train-way 30 --test-way 5 --save-path ./save/proto-5-change --gpu 0
```

For the 5-way 1-shot:
```bash
python train.py --max-epoch 200 --save-epoch 20 --shot 1 --query 10 --train-way 20 --test-way 5 --save-path ./save/proto-1-change --gpu 0
```

## Comparation

![MiniImageNet](assets/table1.png)

![FC100](assets/table2.png)

### cross domain
![CUB](assets/table3.png)

## Citation
If you use this repository in your work, please cite the following paper:
```bibtex
@article{askari2024enhancing,
  title={Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms},
  author={Askari, Fatemeh and Fateh, Amirreza and Mohammadi, Mohammad Reza},
  journal={arXiv preprint arXiv:2409.07989},
  year={2024}
}
