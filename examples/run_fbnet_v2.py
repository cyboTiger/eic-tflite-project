#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example code to run fbnet model on a given image

Usage:
    python3 -m examples.run_fbnet_v2

"""

import urllib

import torch
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess
from PIL import Image
import os


def _get_input():
    # Download an example image from the pytorch website
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/dog.jpg",
        "dog.jpg",
    )
    local_filename, headers = urllib.request.urlretrieve(url, filename)
    input_image = Image.open(local_filename)
    return input_image


def run_fbnet_v2():
    # fbnet models, supported models could be found in
    # mobile_cv/model_zoo/models/model_info/fbnet_v2/*.json
    model_name = "fbnet_a"

    # load model
    model = fbnet(model_name, pretrained=True)
    model.eval()
    preprocess = get_preprocess(model.arch_def.get("input_size", 224))

    # load and process input
    img_dir = '/root/autodl-tmp/imagenet-1k'
    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)
    

    # run model
    with torch.no_grad():
        for img in img_list[:2]:
            input_image = Image.open(os.path.join(img_dir, img))
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            output = model(input_batch)

            output_softmax = torch.nn.functional.softmax(output[0], dim=0)

            img_id = img.split('_')[-1].split('.')[0]
            print(img_id)
            print(output_softmax.argmax(0).item())
            print()


if __name__ == "__main__":
    run_fbnet_v2()
