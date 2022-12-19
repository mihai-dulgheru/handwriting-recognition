# Handwriting recognition

Training a handwriting recognition model with variable-length sequences.

```bash
wget -q https://git.io/J0fjL -O IAM_Words.zip
unzip -qq IAM_Words.zip
mkdir data
mkdir data/words
tar -xf IAM_Words/words.tgz -C data/words
mv IAM_Words/words.txt data
```