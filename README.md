# Siren - Sinusoidal Representation Networks

Unofficial implementation of 'Implicit Neural Representations with Periodic Activation Functions' using **pure pytorch** for the model and fastai2 for it's amazing features to load the data and training loop implementing best pratices.


Video: https://www.youtube.com/watch?v=Q2fLWGBeaiI

Original project page: https://vsitzmann.github.io/siren/

Paper: https://arxiv.org/abs/2006.09661

Only the image and audio fitting experiments from the paper were reproduced here.
Adittionally, an extra experiment for image classification is also available (imagenette notebook).

## Differences between the paper and this implementation

* The model I used is smaller than the one used on the paper, they use 5 layers with a hidden layer size of 256, while I achieved similar results using 5 layers but with cascading sizes of [256, 128, 64, 32]

* The optimizer is upgraded from Adam to [Ranger](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d), and so was the learning rate scheduling

* In the original paper the factor W0 only multiplies the matrix W of the linear layer. I tested it both this way and multiplying the whole result of the linear layer, with no difference in results.
So, to make it easier to compose the activation with different layers I opted for refactoring so that a Siren activation computes:
`Siren(x) = sin(W0 * x)`, making the usage similar to other activations.



## How to run

If you only want to use the activation or model, the code is present at the file `siren.py` and it's **pure pytorch**.
Fastai is used to load the data and training loop during the experiments.

First install fastai2 according to the official instructions present on the repo: https://github.com/fastai/fastai2

Optionally, to calculate and visualize the image gradients the amazing Kornia library is used, check it out: https://kornia.github.io/

If you want to run the audio example, fastai2_audio is also necessary. Instructions to install are present at: https://github.com/rbracco/fastai2_audio

Each notebook represents one experiment.

Files used to train were the first ones that I could find on my machine. The audio is part of the LAPSBM dataset https://gitlab.com/fb-audio-corpora/lapsbm16k while the image is part of The Oxford-IIIT Pet Dataset https://www.robots.ox.ac.uk/~vgg/data/pets/

