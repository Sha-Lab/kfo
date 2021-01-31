
GPU=0

all:
	python examples/anil-kfo.py --help

anil-cfs:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	python examples/anil-kfo.py \
	    --dataset='cifarfs' \
	    --layers=2 \
	    --fast-lr=0.1 \
	    --meta-lr=0.001 \
	    --num-iterations=20000 \
	    --seed=1234

anil-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	python examples/anil-kfo.py \
	    --dataset='mini-imagenet' \
	    --layers=2 \
	    --fast-lr=0.05 \
	    --meta-lr=0.001 \
	    --num-iterations=3500 \
	    --seed=1234

maml-cfs:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	python examples/maml-kfo.py \
	    --dataset='cifarfs' \
	    --layers=2 \
	    --fast-lr=0.1 \
	    --meta-lr=0.0005 \
	    --num-iterations=20000 \
	    --seed=1234

maml-mi:
	CUDA_VISIBLE_DEVICES=$(GPU) \
	python examples/maml-kfo.py \
	    --dataset='mini-imagenet' \
	    --layers=2 \
	    --fast-lr=0.001 \
	    --meta-lr=0.0005 \
	    --num-iterations=20000 \
	    --seed=1234
