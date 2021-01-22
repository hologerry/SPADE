"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import datasets
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
# from util.visualizer import Visualizer
from util import html
from os.path import join as ospj
from torchvision.utils import save_image

opt = TestOptions().parse()

dataloader = datasets.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

# visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))
exper_dir = ospj(opt.checkpoints_dir, opt.name)
result_dir = ospj(exper_dir, "results")
os.makedirs(result_dir, exist_ok=True)


# test
for i, data_i in enumerate(dataloader):
    # if i * opt.batchSize >= opt.how_many:
    #     break

    generated = model(data_i, mode='inference')
    save_file = ospj(result_dir, f"test_epoch_{opt.which_epoch}_{opt.dataset_name}_batch_{i:03d}.png")
    save_image(generated, save_file, nrow=generated.size(0), normalize=True, padding=0)
    # img_path = data_i['path']
    # for b in range(generated.shape[0]):
    #     print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('input_label', data_i['label'][b]),
    #                            ('synthesized_image', generated[b])])
        # visualizer.save_images(webpage, visuals, img_path[b:b + 1])

# webpage.save()
