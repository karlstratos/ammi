# Adversarial Maximal Mutual Information (AMMI)
This is a PyTorch implementation of AMMI [1]. Install dependencies (Python 3) and download data by running 
```
pip install -r requirements.txt
./get_data.sh
```
Specifically, the experiments were run with Python `3.8.3` and PyTorch `1.5.3` using NVIDIA Quadro RTX 6000s (CUDA version `10.2`).

### Quick start
**Unsupervised document hashing** on Reuters using 16 bits
```bash
python ammi.py reuters16_ammi data/document_hashing/reuters.tfidf.mat --train --raw_prior
```
Output logged in file `reuters16_ammi.log`. You can simply switch the dataset to do **predictive document hashing**, for instance, 
```bash
python ammi.py toy data/related_articles/article_pairs_tfidf_small.p --train --raw_prior --num_retrieve 10
```
The VAE and DVQ baselines can be run similarly by switching `ammi.py` with `vae.py` or `dvq.py`.

### Reproducibility
See [`commands.txt`](commands.txt) for the hyperparameters used in the paper. They were optimized by random grid search on validation data, for instance 
```bash
python ammi.py tmc64_ammi data/document_hashing/reuters.tfidf.mat --train --num_features 64 --num_runs 100 --cuda 
python ammi.py wdw128_ammi data/related_articles/article_pairs_tfidf.p --train --num_features 128 --num_runs 100 --cuda --num_workers 8
```

### References
[1] [Learning Discrete Structured Representations by Adversarially Maximizing Mutual Information (Stratos and Wiseman, 2020)](https://arxiv.org/abs/2004.03991)
```
@article{stratos2020learning,
  title={Learning Discrete Structured Representations by Adversarially Maximizing Mutual Information},
  author={Stratos, Karl and Wiseman, Sam},
  journal={arXiv preprint arXiv:2004.03991},
  year={2020}
}
```
