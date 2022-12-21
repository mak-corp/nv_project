# #!/bin/bash

# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# cd $SCRIPT_DIR

# echo
# echo "===================== Setup started ====================="

# mkdir -p data
# cd data

# # Texts
# echo 'Download texts...'
# python3 -c '
# import gdown
# gdown.download("https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx")
# '
# mv train.txt data/
# echo 'Complited'

# # Mels
# echo 'Download mels...'
# python3 -c '
# import gdown
# gdown.download("https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j")
# '
# echo 'Unpacking mels...'
# tar -xf mel.tar.gz >> /dev/null
# echo 'Complited'

# # Alignments
# echo 'Download alignments...'
# wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
# echo 'Unpacking alignments...'
# unzip alignments.zip >> /dev/null
# echo 'Complited'

# # Waveglow
# echo 'Download WaveGlow...'
# python3 -c '
# import gdown
# gdown.download("https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx")
# '
# mv waveglow_256channels_ljs_v2.pt waveglow_256channels.pt
# echo 'Complited'

# # LJSpeech
# echo 'Download LJSpeech...'
# wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
# echo 'Unpacking LJSpeech...'
# tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
# echo 'Complited'

# cd ../

# # Preprocess audio
# echo 'Preprocess LJSpeech...'
# python3 audio_processing.py --ljspeech-dir data/LJSpeech-1.1 --output-dir data_v2
# echo 'Complited'

# # Download checkpoints
# mkdir checkpoints
# echo 'Download checkpoints...'
# python3 -c '
# import yadisk
# y = yadisk.YaDisk()
# y.download_public("https://disk.yandex.ru/d/nuo4pvUqQnoryA", "checkpoints/checkpoints.zip")
# '
# echo 'Unpacking checkpoints...'
# cd checkpoints
# unzip checkpoints.zip

# echo 'Complited'


# echo
# echo "===================== Setup complited ====================="
# echo