import io

import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor.transpose(-1, -2))
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close('all')
    buf.seek(0)
    return buf
