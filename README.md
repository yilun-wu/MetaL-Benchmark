# MetaL-Benchmark
Accompanying Code for "ALPaCA vs. GP-based Prior Learning: A comparison between two Bayesian Meta-Learning Algorithms". [Link](TBD)

The report investigates similarites and disparities among two recently published Bayesian meta-learning methods: 
[ALPaCA (Harrison et al., 2018)](https://arxiv.org/abs/1807.08912) and [PACOH (Rothfuss et al., 2020)](https://arxiv.org/abs/2002.05551).
Theoratical analysis as well as empricial benchmarks (produced by the code in this repo) are presented.
## Installation
After cloning the repository, first checkout the submodules:
```
git submodule update --init --recursive
```
Then to install requirements, run
```
pip install -r lib/ALPaCA/requirements.txt lib/PACOH/requirements.txt
```


## Running the code
The experiments presented in the paper can be run from the jupyter notebook under `src/exp-supervised-benchmark.ipynb`
The script will save the trained models in a folder name hashed by experiment hyper-parameters under `data/`. 
The experiment runs presented in the paper are with the following hashes available at https://www.polybox.ethz.ch/index.php/s/fSqUUCSjfyjcUt8.

|  Model             |  Sinusoid-E                       |          |         |         |  Sinusoid-H                        |          |         |         |  Cauchy                            |          |         |         |  Swissfel                          |          |         |         |
|--------------------|-----------------------------------|----------|---------|---------|------------------------------------|----------|---------|---------|------------------------------------|----------|---------|---------|------------------------------------|----------|---------|---------|
|                    |  Hash                             |  LL      |  RMSE   |  Calib  |  Hash                              |  LL      |  RMSE   |  Calib  |  Hash                              |  LL      |  RMSE   |  Calib  |  Hash                              |  LL      |  RMSE   |  Calib  |
|  GP+SE+NN          |  556ed7df63f1b672038567f03681f498 |  0.313   |  0.315  |  0.120  |  afd05d82a75fe279cfc0785d70abcfbe  |  -0.112  |  0.644  |  0.108  |  d51b8f751778e3805a91dc9f20a8bd7a  |  0.394   |  0.200  |  0.060  |  dcd464ceee4071a362a3cd52b93d57e6  |  -0.447  |  0.368  |  0.086  |
|  GP+NN+NN          |  b234101b8728569befd823ffc45ffcc2 |  0.596   |  0.287  |  0.124  |  b8306037d50693b96a93d79b2952546a  |  -0.125  |  0.632  |  0.108  |  e6d5c1f43644b42b69d456068ed1f5d2  |  0.185   |  0.217  |  0.069  |  b141d967a1bc4e14fc444f42aaee220f  |  -0.763  |  0.443  |  0.057  |
|  GP+NNL+NN         |  afed0fa8f682754c46cc6870eebef424 |  0.122   |  0.248  |  0.130  |  e212f5e2efa7e62b3034d970cdbae158  |  -1.056  |  0.743  |  0.110  |  4a17c486fda36a836fa95c8102057414  |  -0.015  |  0.239  |  0.074  |  bbd17d55149b77eb8d4f7b1bd340fcbf  |  -1.228  |  0.663  |  0.076  |
|  GP+NNL+NNOne      |  e90150fdac160c6488d6f5900dddcce  |  0.141   |  0.218  |  0.142  |  47bd412edfd12b41d40a5169399af0e6  |  -1.204  |  0.863  |  0.100  |  0de7c4f47bbddbbe92e937817c48aa54  |  0.016   |  0.230  |  0.076  |  d420c137b59ad709f5272f6bd4aa65b6  |  -0.645  |  0.459  |  0.054  |
|  BLR-Prior-Full    |  584e9c6f3963f0b2a4626a14f54c8acd |  -0.203  |  0.340  |  0.118  |  51edfcd7353c74261817cd06f40643be  |  -1.203  |  0.884  |  0.100  |  c10e083c86229b284fc0ce9b374024df  |  0.011   |  0.225  |  0.078  |  760a8ed714377a28e6d1e276e73255ae  |  -0.826  |  0.479  |  0.074  |
|  BLR-Prior-NF      |  6f3ae0ef48fb343b17233554d9f49cc2 |  -1.21   |  0.748  |  0.173  |  a80d44b16e63645932be9dcd1fd810e5  |  -1.302  |  0.949  |  0.102  |  6fa8a0c3a0ce603dc74ea87f64c7fa18  |  -0.308  |  0.237  |  0.112  |  a0a49a393d1a13463cb5b99de4d85762  |  -1.768  |  0.641  |  0.146  |
|  BLR-Post-Full-C   |  5fc3c5d32a28895182f9a22b4bb1b106 |  -0.45   |  0.438  |  0.111  |  0e80925297c4477f60ba960369ac9b14  |  -1.266  |  0.919  |  0.096  |  e74b047ea2dbaa23f69ef93c495c49ec  |  0.044   |  0.231  |  0.075  |  b047ed2052679791873486d582434e9c  |  -0.979  |  0.630  |  0.078  |
|  BLR-Post-Full-NC  |  c2283feb46551cd0601f0b04951b321a |  -0.373  |  0.404  |  0.116  |  faf9956480fbcfbaa509906ba796b703  |  -1.226  |  0.905  |  0.100  |  ed0b5b9fdf2c040cf6142527f9b7dc03  |  -0.038  |  0.246  |  0.080  |  ac57d7f4fa73259bd39a0aa227f7bed7  |  -1.892  |  0.828  |  0.139  |
|  BLR-Post-NF-NC    |  756a425c5c6921b89e6f804314c272fd |  -0.587  |  0.481  |  0.132  |  73a1e72c5c6ac8a7fb727502de72a32d  |  -1.264  |  0.944  |  0.098  |  bafd4b87d2abb196f8d4479aca329e48  |  -0.193  |  0.234  |  0.102  |  3afd3884aa170b49c929c8d0cc0c385f  |  -1.406  |  0.967  |  0.143  |


## Note
The Swissfel dataset has not been made publicly available.