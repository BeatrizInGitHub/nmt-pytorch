# Neural Machine Translation Pytorch Implementation
A Pytorch Implementation of paper
> [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) <br>
> Bahdanau et al., 2015 ICLR

## Requirements
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Pytorch 0.4.0](https://pytorch.org/)
- Python version >= 3.5 is required
- TorchText + Spacy is required for preprocessing

## Datasets
- WMT14 downloaded from [here](http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/)
- Place files under (home)/datasets/wmt14/(train/valid/test) directory.

```bash
# Preprocessing dataset. This will create ./data/wmt14(30).pkl
$ python dataset.py
```

## Run experiments
```bash
# Train and test with default settings (Seq2SeqAttModel)
$ python main.py

# Train with different number of rnn hidden units and epochs
$ python main.py --rnn-dim 500 --epoch 10
```
- Refer to the [paper](https://arxiv.org/abs/1409.0473) for more detailed explanations of the model.

## Results (vs RNNsearch-30)
| Reported BLEU | Our Implementation (valid) | Our Implementation (test) |
|:-------------:|:--------------------------:|:-------------------------:|
|     92.3      |           71.1             |           108.9           |

## Licence
MIT
