# When MAML Can Adapt Fast and How to Assist When It Cannot

[![AISTATS](https://img.shields.io/badge/AISTATS-2021-informational.svg)](http://seba1511.net/projects/kfo/)

Code release for "When MAML Can Adapt Fast and How to Assist When It Cannot", AISTATS 2021.

This code provides a re-implementation of the MAML-KFO and ANIL-KFO algorithm in `examples/`.

## Resources

* Website: [seba1511.net/projects/kfo](http://seba1511.net/projects/kfo)
* Preprint: [arxiv.org/abs/1910.13603](https://arxiv.org/abs/1910.13603)
* Code: [github.com/Sha-Lab/kfo](https://github.com/Sha-Lab/kfo)

## Citation

Please cite this work as follows:

> Sébastien M. R. Arnold, Shariq Iqbal and Fei Sha, "When MAML Can Adapt Fast and How to Assist When It Cannot". AISTATS 2021.

or with the following BibTex entry:

~~~bibtex
@inproceedings{Arnold2021MAML,
    title={When MAML Can Adapt Fast and How to Assist When It Cannot},
    author={S\'ebastien M. R. Arnold and Shariq Iqbal and Fei Sha},
    booktitle={AISTATS},
    year={2021},
    note={to appear},
}
~~~

## Usage

Dependencies include the following Python packages:

* PyTorch>=1.3.0
* torchvision>=0.5.0
* scikit-learn>=0.19.2
* tqdm>=4.48.2
* learn2learn on the master branch

To create a conda environment (named `kfo`) with all dependencies, run: `conda env create -f environment.yaml`

### Running Experiments

We provide example implementations for ANIL+KFO and MAML+KFO.
To run those examples, use:

~~~shell
make [anil-cfs | anil-mi | maml-cfs | maml-mi]
~~~

where each target is described in the `Makefile`.

**Note:** For simplicity, those implementations use a linear Kronecker-factored meta-optimizer.
Some results in the paper use non-linear architectures, typically with 2 hidden layers.

For more information about the command line interface, please run:

~~~shell
python examples/anil-kfo.py --help

or:

python examples/maml-kfo.py --help
~~~
