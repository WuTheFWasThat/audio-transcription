# Setup

- install tensorflow
- install python library requirements
  ```
  pip install matplotlib
  pip install pyknon
  pip install scipy
  pip install sh
  ```
- for osx (dunno about others)
  ```
  brew install lame
  brew install timidity
  ```

# Run

- to train:

  `python train.py`

- to see summaries while training:

  `tensorboard --logdir=summaries/model`

- debug a model:

  `python debug.py --checkpoint 2200`.

  on os x, you can then `afplay $filename` to hear the file.
