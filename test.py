# import argparse
# import json
# import os
# from pathlib import Path

# import torch
# from tqdm import tqdm

# import numpy as np
# from scipy.io.wavfile import write

# import nv_lib.model as module_model
# from nv_lib.trainer import Trainer
# from nv_lib.utils import ROOT_PATH
# from nv_lib.utils.object_loading import get_dataloaders
# from nv_lib.utils.parse_config import ConfigParser
# from nv_lib.mel_2_wav import WaveGlowInfer

# from nv_lib.datasets.test_data import get_test_data, get_v1_test_data

# DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
# MAX_WAV_VALUE = 32768.0


# def main(config, out_dir):
#     logger = config.get_logger("test")

#     # define cpu or gpu if possible
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # build model architecture
#     model = config.init_obj(config["arch"], module_model, config["arch"]["args"])
#     logger.info(model)

#     logger.info("Loading checkpoint: {} ...".format(config.resume))
#     checkpoint = torch.load(config.resume, map_location=device)
#     state_dict = checkpoint["state_dict"]
#     if config["n_gpu"] > 1:
#         model = torch.nn.DataParallel(model)
#     model.load_state_dict(state_dict)

#     # prepare model for testing
#     model = model.to(device)
#     model.eval()

#     if config["arch"]["type"] == "FastSpeechBaselineModel":
#         test_data = get_v1_test_data()
#     else:
#         test_data = get_test_data()

#     waveglow = WaveGlowInfer(config["waveglow_path"], device)

#     with torch.no_grad():
#         for text_id, batches in tqdm(test_data.items()):
#             for batch in tqdm(batches):
#                 batch = Trainer.move_batch_to_device(batch, device)
#                 output_mel = model(**batch)
#                 batch["output_mel"] = output_mel

#                 audio, sr = waveglow(output_mel[0])
#                 audio = audio * MAX_WAV_VALUE
#                 audio = audio.cpu().numpy()
#                 audio = audio.astype('int16')

#                 name = "{}_text={}.wav".format(batch["name"], batch["text_id"])
#                 write(os.path.join(out_dir, name), sr, audio)


# if __name__ == "__main__":
#     args = argparse.ArgumentParser(description="PyTorch Template")
#     args.add_argument(
#         "-c",
#         "--config",
#         default=None,
#         type=str,
#         help="config file path (default: None)",
#     )
#     args.add_argument(
#         "-r",
#         "--resume",
#         default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
#         type=str,
#         help="path to latest checkpoint (default: None)",
#     )
#     args.add_argument(
#         "-d",
#         "--device",
#         default=None,
#         type=str,
#         help="indices of GPUs to enable (default: all)",
#     )
#     args.add_argument(
#         "-o",
#         "--output",
#         type=str,
#         help="Folder to write results",
#     )

#     args = args.parse_args()

#     # set GPUs
#     if args.device is not None:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.device

#     # first, we need to obtain config with model parameters
#     # we assume it is located with checkpoint in the same folder
#     model_config = Path(args.resume).parent / "config.json"
#     with model_config.open() as f:
#         config = ConfigParser(json.load(f), resume=args.resume)

#     # update with addition configs from `args.config` if provided
#     if args.config is not None:
#         with Path(args.config).open() as f:
#             config.config.update(json.load(f))

#     os.makedirs(args.output, exist_ok=True)
#     main(config, args.output)
