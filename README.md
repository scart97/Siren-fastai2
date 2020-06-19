# Siren-fastai2
Unofficial implementation of 'Implicit Neural Representations with Periodic Activation Functions' using fastai2

Original paper can be found here: https://vsitzmann.github.io/siren/

Only the image and audio fitting experiments were reproduced here.

## How to run

First install fastai2 according to the official instructions present on the repo: https://github.com/fastai/fastai2

If you want to run the audio example, fastai2_audio is also necessary. Instructions to install are present at: https://github.com/rbracco/fastai2_audio

The file `siren.py` contains the model, and each notebook represents one experiment.

Files used to train were the first ones that I could find on my machine. The audio is part of the LAPSBM dataset https://gitlab.com/fb-audio-corpora/lapsbm16k while the image is part of The Oxford-IIIT Pet Dataset https://www.robots.ox.ac.uk/~vgg/data/pets/

