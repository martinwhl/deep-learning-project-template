# deep-learning-project-template

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/deep-learning-project-template?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/deep-learning-project-template/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/deep-learning-project-template)](https://github.com/martinwhl/deep-learning-project-template/issues) [![License](https://img.shields.io/github/license/martinwhl/deep-learning-project-template)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/deep-learning-project-template/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/deep-learning-project-template/badge)

This is a deep learning project template designed for PyTorch and PyTorch Lightning users.

## How to use this template

1. Run the following commands...

```bash
pip install -r requirements-dev.txt
git init
pre-commit install  # automatically format and do style check when committing
```

2. Start coding!

The files and directories are organized as follows:
```
- data
    |- ...  # put your data here, do not include large files in git
- models
    |- ...  # models should be `torch.nn.Module`
- tasks
    |- ...  # tasks should be a framework or a system inherited from `pl.LightningModule`
- utils
    |- callbacks    # put your custom PyTorch Lightning callbacks here
    |- data         # put `torch.utils.data.Dataset` and `pl.LightningDataModule` subclasses here
    |- metrics      # put your custom metrics here, it is recommended to inherit from `torchmetrics.Metric`
```
An MLP classifier and an MNIST data module are already implemented as examples. Replace them with your custom tasks/models/data modules.

## Requirements

* torch
* torchvision (if you want to run the MNIST example, otherwise unnecessary)
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* pre-commit
* black
* flake8
