#!/bin/sh
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove.6B
unzip glove.6B.zip -d glove.6B
if [ $? -eq 0 ]; then
    rm glove.6B.zip
else
    echo 'Unzipping Failed'
fi
pip install -U textblob
python -m textblob.download_corpora lite