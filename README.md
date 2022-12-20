# Handwriting recognition

Training a handwriting recognition model with variable-length sequences.

## Introduction

This example shows how the [Captcha OCR](https://keras.io/examples/vision/captcha_ocr/)
example can be extended to the
[IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database),
which has variable length ground-truth targets. Each sample in the dataset is an image of some
handwritten text, and its corresponding target is the string present in the image.
The IAM Dataset is widely used across many OCR benchmarks, so we hope this example can serve as a
good starting point for building OCR systems.

## Data collection

```bash
wget -q https://git.io/J0fjL -O IAM_Words.zip
unzip -qq IAM_Words.zip
mkdir data
mkdir data/words
tar -xf IAM_Words/words.tgz -C data/words
mv IAM_Words/words.txt data
```