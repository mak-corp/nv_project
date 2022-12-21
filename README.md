# NV project

In this project I've implemented HiFiGAN model to generate waveforms from melspectrograms.

WandB report:
https://wandb.ai/mak_corp/nv_project/reports/NV-project-report--VmlldzozMTk2MTA5


Installation guide:
```shell

chmod +x setup.sh
./setup.sh

```

Run train:
```shell

python3 train.py -c nv_lib/configs/hifigan.json

```

Run test:
```shell

python3 test.py -r checkpoints/NV/best_model/model_best.pth -o generated

```

Full task description is available here:
https://github.com/markovka17/dla/tree/2022/hw4_nv
