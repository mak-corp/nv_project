import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import numpy as np
from scipy.io.wavfile import write
import torchaudio

import nv_lib.model as module_model
from nv_lib.model import MelSpectrogramConfig
from nv_lib.trainer import Trainer
from nv_lib.utils import ROOT_PATH
from nv_lib.utils.object_loading import get_dataloaders
from nv_lib.utils.parse_config import ConfigParser

MAX_WAV_VALUE = 32768.0


def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_extra_state_dict(checkpoint["extra_state_dict"])

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloaders["test"]), start=1):
            batch = Trainer.move_batch_to_device(batch, device)
            batch = model(batch)

            torchaudio.save(
                os.path.join(out_dir, f"gen_audio_{batch_idx}.wav"),
                batch["gen_wav"],  # batch dim is like channel dim
                MelSpectrogramConfig.sr
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder to write results",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    os.makedirs(args.output, exist_ok=True)
    main(config, args.output)
