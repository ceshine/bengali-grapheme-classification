# TPU-Ready TF 2.1 Solution to Bengali.AI Handwritten Grapheme Classification

A rather generic image classification pipeline with TPU-compatible MixUp and CutMix implementations.

(This is the preliminary documentation. Please create an issue if you have any specific questions.)

## Preparation

Install Tensorflow >= 2.1 and run `pip install tf-helper-bot/.`.

### Build the wheels

`python setup.py sdist bdist_wheel`

And upload the `.whl` files in the `dist` directory to Google Cloud Storage.

#### Create the TFRecord files

Create data splits:

`python scripts/split_train.py`

Prepare TFRecord files (the following only prepares the data for one fold):

`python -m bengali.prepare_tfrecords data/train_split_0.csv data/tfrecords/train_0/`

`python -m bengali.prepare_tfrecords data/valid_split_0.csv data/tfrecords/valid_0/`

Upload the content in `data/tfrecords` to Google Cloud Storage:

(Note: check [requirements.txt](requirements.txt) for missing dependencies.)

## Training

Example command:

`python -m bengali.train --batch-size 64 --arch b4 --checkpoint-interval 2000 --steps 20010 --train-folder "gs://ceshine-tpu-us-central/bengali/train_0/*" --valid-folder "gs://ceshine-tpu-us-central/bengali/valid_0/*" --resize 192,330 --max-lr 4e-3 --mixup-alpha -1 --cutmix-alpha 1. --weight-decay 0 --log-interval 500 --mixed-precision --output-suffix _192330_0 --radam`
