import argparse
import pytorch_lightning as pl
import models
import tasks
import utils.callbacks
import utils.data
import utils.misc
from pytorch_lightning.utilities import rank_zero_info


def main(args):
    rank_zero_info(vars(args))

    dm = getattr(utils.data, args.data + "DataModule")(**vars(args))
    dm.prepare_data()
    dm.setup()
    model = getattr(models, args.model_name)(
        input_dim=dm.num_pixels, output_dim=dm.num_classes, **vars(args)
    )
    task = getattr(tasks, args.task.capitalize() + "Task")(backbone=model, **vars(args))

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[pl.callbacks.ModelCheckpoint(monitor="Val_Loss")]
    )
    trainer.fit(task, dm)
    trainer.test(task, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data",
        type=str,
        default="MNIST",
        choices=("MNIST", "CIFAR10"),
        help="The name of the dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MLP",
        choices=("MLP", "AlexNet", "VGG19", "ResNet50"),
        help="The name of the model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=("classification"),
        help="The type of the task",
    )
    parser.add_argument(
        "--log_path", type=str, default=None, help="Path to the output console log file"
    )

    temp_args, _ = parser.parse_known_args()

    parser = getattr(
        utils.data, temp_args.data + "DataModule"
    ).add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(
        tasks, temp_args.task.capitalize() + "Task"
    ).add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.misc.format_logger(pl._logger)
    if args.log_path is not None:
        utils.misc.output_logger_to_file(pl._logger, args.log_path)

    main(args)
