#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

echo
echo "===================== Setup started ====================="

echo 'Install Dependencies...'
python3 -m pip install -r requirements.txt
echo 'Complited'

mkdir -p data
cd data

# LJSpeech
echo 'Download LJSpeech...'
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
echo 'Unpacking LJSpeech...'
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
echo 'Complited'

cd ../

# Download checkpoints
mkdir checkpoints
echo 'Download checkpoints...'
python3 -c '
import yadisk
y = yadisk.YaDisk()
y.download_public("https://disk.yandex.ru/d/8G5A4e4FOcBzbQ", "checkpoints/checkpoints.zip")
'
echo 'Unpacking checkpoints...'
cd checkpoints
unzip checkpoints.zip

echo 'Complited'


echo
echo "===================== Setup complited ====================="
echo