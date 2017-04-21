# DCNN
An implementation of [Diffusion-Convolutional Neural Networks](http://papers.nips.cc/paper/6212-diffusion-convolutional-neural-networks.pdf) [1] in Theano and Lasagne.

## Installation
    git clone https://github.com/jcatw/dcnn.git
	cd dcnn

## Usage
### Node Classification (Cora)
	python -m client.run --model=node_classification --data=cora
### Graph Classification (NCI1)
	python -m client.run --model=graph_classification --data=nci1

# References
[1] [Atwood, James, and Don Towsley. "Diffusion-convolutional neural networks." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6212-diffusion-convolutional-neural-networks.pdf)

