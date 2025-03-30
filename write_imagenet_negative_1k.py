from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

from negativeDataset import NegativeDataset

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

import os

Section("cfg", "arguments to give the writer").params(
    dataset=Param(
        And(str, OneOf(["cifar", "imagenet"])),
        "Which dataset to write",
        default="imagenet",
    ),
    split=Param(And(str, OneOf(["train", "val"])), "Train or val set", required=True),
    negative_path=Param(str, "Where to find the PyTorch dataset", required=True),
    positive_path=Param(str, "Where to find the PyTorch dataset", required=True),
    write_mode=Param(str, "Mode: raw, smart or jpg", required=False, default="smart"),
    max_resolution=Param(int, "Max image side length", required=True),
    num_workers=Param(int, "Number of workers to use", default=16),
    chunk_size=Param(int, "Chunk size for writing", default=100),
    jpeg_quality=Param(float, "Quality of jpeg images", default=90),
    subset=Param(int, "How many images to use (-1 for all)", default=-1),
    compress_probability=Param(float, "compress probability", default=None),
    neg_counts=Param(int, "Number of negative samples per class", default=75),
    write_path=Param(str, "name", default=None),
    superset_file_src=Param(str, "file with the pool of negative samples to sample from", default=None),
    subset_file_dst=Param(str, "file where to write the pool of negative samples to sample from", default=None),
)

def _get_beton_path(
    negative_path,
    max_resolution,
    compress_probability,
    jpeg_quality,
    neg_counts,
    subset
):
    "Get the path to the ffcv files."
    sufix = ""
    if subset > 0:
        sufix = "debug"
    template = f"{'{}'}_{max_resolution}_{compress_probability}_{jpeg_quality}"
    training_path = template.format(negative_path) + f"_{neg_counts}" + f"{sufix}"
    return training_path + ".beton"

@section("cfg")
@param("dataset")
@param("split")
@param("negative_path")
@param("positive_path")
@param("max_resolution")
@param("num_workers")
@param("chunk_size")
@param("subset")
@param("jpeg_quality")
@param("write_mode")
@param("compress_probability")
@param("neg_counts")
@param("write_path")
@param("superset_file_src")
@param("subset_file_dst")
def maybe_write_ffcv(
    dataset,
    split,
    negative_path,
    positive_path,
    max_resolution,
    num_workers,
    chunk_size,
    subset,
    jpeg_quality,
    write_mode,
    compress_probability,
    neg_counts,
    write_path,
    superset_file_src,
    subset_file_dst
):
    dst = write_path
    if not dst:
        dst = _get_beton_path(
            negative_path,
            max_resolution,
            compress_probability,
            jpeg_quality,
            neg_counts,
            subset
        )
    if os.path.exists(dst):
        print("Dataset already exists, skipping.")
        return

    print("Writing FFCV")
    if dataset == "imagenet":       
        my_dataset = NegativeDataset(
            root="",
            root_pos=positive_path,
            root_neg=negative_path,
            class_sample_counts=neg_counts,
            negative_samples_file=superset_file_src,
            sampling_method='sorted'
        )
        my_dataset.save_negative_samples_to_file(subset_file_dst)
        print(f"Written file to {subset_file_dst}")
    else:
        raise ValueError("Unrecognized dataset", dataset)

    if subset > 0:
        print("Using subset")
        my_dataset = Subset(my_dataset, range(1000))
    writer = DatasetWriter(
        dst,
        {
            "image": RGBImageField(
                write_mode=write_mode,
                max_resolution=max_resolution,
                compress_probability=compress_probability,
                jpeg_quality=jpeg_quality,
            ),
            "label": IntField(),
            "charge": IntField(),
            # "index": IntField(),
        },
        num_workers=num_workers,
    )
    print("start")
    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)
    print("Finished writing FFCV")

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    maybe_write_ffcv()
