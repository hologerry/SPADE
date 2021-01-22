import os
from os.path import join as ospj

import torch.utils.data as data
from PIL import Image

from datasets.base_dataset import BaseDataset
from datasets.dataset_utils import arbitrary_aspect_ratio_hd, arbitrary_reference_fetch, random_crop_style, random_flip_hd, transform_references
from datasets.reference_hd_dataset import get_image_transform

class ReferenceTestHDDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, opt):
        self.data_root = opt.dataroot
        self.dataset_name = opt.dataset_name
        self.dataset_dir = 'reference_hd'
        self.case_name = opt.case_name
        self.trans = get_image_transform()
        self.images_dir = ospj(opt.dataroot, self.dataset_dir, opt.case_name, 'art')
        # if mode == 'test':
        #     sketch_dir = 'test'
        # else:
        #     raise ValueError
        sketch_dir = opt.mode

        self.images_files = sorted(os.listdir(self.images_dir))
        self.sketches_dir = ospj(opt.dataroot, self.dataset_dir, opt.case_name, sketch_dir)
        self.images_files = sorted(os.listdir(self.images_dir))
        self.sketches_files = sorted(os.listdir(self.sketches_dir))
        self.num_images = len(self.sketches_files)

        self.length = self.num_images

    def __getitem__(self, index):
        img_idx = int(index % len(self.images_files))
        img_path = ospj(self.images_dir, self.images_files[img_idx])
        skt_path = ospj(self.sketches_dir, self.sketches_files[index])
        # img = Image.open(img_path).convert('RGB')
        skt = Image.open(skt_path).convert('RGB')
        skt_w, skt_h = skt.size
        img_none = Image.new('RGB', (skt_w, skt_h), 'white')  # just for placeholding
        img_hd_none = Image.new('RGB', (skt_w, skt_h), 'white')  # just for placeholding
        # style_img = random_crop_style(img, self.style_size)
        # reference_bank = arbitrary_reference_fetch(self.data_root, self.dataset_dir, self.case_name, self.reference_num, self.sample_per_ref)
        # skt_size = skt.size
        # ref_bank = transform_references(reference_bank, skt_size, self.trans)
        return {
            "label": self.trans(skt), "image": self.trans(img_none),
        }

    def __len__(self):
        return self.length
