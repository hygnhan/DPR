##  Mitigating Spurious Correlations via Disagreement Probability
This repository provides the official PyTorch implementation of the following paper:
> Mitigating Spurious Correlations via Disagreement Probability<br>
> Hyeonggeun Han, Sehwan Kim, Hyungjun Joo, Sangwoo Hong, Jungwoo Lee<br>
> *Seoul National University*<br>
>  

> Paper: [Arxiv](https://arxiv.org/abs/2411.01757) <br>

**Abstract:** 
*Models trained with empirical risk minimization (ERM) are prone to be biased towards spurious correlations between target labels and bias attributes, which leads to poor performance on data groups lacking spurious correlations. It is particularly challenging to address this problem when access to bias labels is not permitted. To mitigate the effect of spurious correlations without bias labels, we first introduce a
novel training objective designed to robustly enhance model performance across all data samples, irrespective of the presence of spurious correlations. From this objective, we then derive a debiasing method, Disagreement Probability based Resampling for debiasing (DPR), which does not require bias labels. DPR leverages the disagreement between the target label and the prediction of a biased model to identify bias-conflicting samples—those without spurious correlations—and upsamples them according to the disagreement probability. Empirical evaluations on multiple benchmarks demonstrate that DPR achieves state-of-the-art performance over existing baselines that do not use bias labels. Furthermore, we provide a theoretical analysis that details how DPR reduces dependency on spurious correlations.*<br>

## Pytorch Implementation

### Constructing datasets
- **Colored MNIST:** 
```
python datagen/generator.py --data colored_mnist --bias_ratio 0.005,0.01,0.05
```

### Running DPR
We demonstrate DPR using the Colored MNIST (0.5%) as an example. To apply the method to other bias-conflicting ratios, simply modify the ```--bratio``` argument accordingly.

#### Colored MNIST (0.5%)
- **Train biased models**
```
python trainer/train.py --reproduce --bratio 0.005 --dataset colored_mnist --train --algorithm dpr --exp run0
```
- **Compute and save sampling probabilities**
```
python trainer/train.py --reproduce --bratio 0.005 --dataset colored_mnist --train --algorithm dpr --pretrained_bmodel --save_sampling_prob --tau 1 --exp run0
```
- **Train debiased models**
```
python trainer/train.py --reproduce --bratio 0.005 --dataset colored_mnist --train --algorithm dpr --pretrained_bmodel --tau 1 --exp run0
```

- **Evaluate debiased models**
```
python trainer/train.py --reproduce --bratio 0.005 --dataset colored_mnist --algorithm dpr --exp run0
```

### Citations

### BibTeX
```bibtex
@article{han2024,
  title={Mitigating spurious correlations via disagreement probability},
  author={Han, Hyeonggeun and Kim, Sehwan and Joo, Hyungjun and Hong, Sangwoo and Lee, Jungwoo},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={74363--74382},
  year={2024}
}
```

### Contact
Hyeonggeun Han (hygnhan@snu.ac.kr)

### Acknowledgments
Our pytorch implementation is based on [PGD](https://github.com/sumyeongahn/PGD/tree/main).
We would like to thank for the authors for making their code public.
