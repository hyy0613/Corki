

<h1 align="center">Corki: Enabling Real-time Embodied AI Robots via Algorithm-Architecture Co-Design</h1>

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)

This is also the official code repo for the paper [Corki: Enabling Real-time Embodied AI Robots via Algorithm-Architecture Co-Design](https://arxiv.org/pdf/2407.04292)

If you have any questions about the paper and code, please contact us.

All our experiments are conducted on a 8 GPUS server with 8 Nvidia A100 GPUs (80G).

## Download the Calvin dataset and models:

Our repository is built based on the work RoboFlamingo . Please follow the [RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo)  to download the corresponding OpenFlamingo model  checkpoints, conda environment, and the Calvin dataset.

## Training the model (using DDP):

```
torchrun --nnodes=1 --nproc_per_node=8  robouniview/train/train.py \
    --config config/robouniview_pretrain.yaml
```

## Evaluating the model on the CALVIN benchmark

```
python eval_ckpts.py
```

## Acknowledgment

#### CALVIN

Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)

#### OpenFlamingo

Original: [https://github.com/mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo)
License: [MIT](https://github.com/mlfoundations/open_flamingo/blob/main/LICENSE)

#### RoboFlamingo

Original: [https://github.com/RoboFlamingo/RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo)
License: [MIT](https://github.com/RoboFlamingo/RoboFlamingo/blob/main/LICENSE)

## Cite our work:

```
@article{huang2024corki,
  title={Corki: Enabling Real-time Embodied AI Robots via Algorithm-Architecture Co-Design},
  author={Huang, Yiyang and Hao, Yuhui and Yu, Bo and Yan, Feng and Yang, Yuxin and Min, Feng and Han, Yinhe and Ma, Lin and Liu, Shaoshan and Liu, Qiang and others},
  journal={arXiv preprint arXiv:2407.04292},
  year={2024}
}
```
