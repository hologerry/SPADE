import os
from os.path import join as ospj

import torch.utils.data as data
from PIL import Image

import torchvision.transforms as T
from datasets.base_dataset import BaseDataset

from datasets.dataset_utils import arbitrary_aspect_ratio, arbitrary_reference_fetch, random_crop_style, random_flip, random_scale_stretch_hd, transform_references

def get_image_transform(mode='train', random_flip_ratio=0.0):
    transforms = []
    # currently not supported
    # if random_flip_ratio > 0.0 and mode == 'train':
    #     transforms.append(T.RandomHorizontalFlip(random_flip_ratio))
    transforms += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return T.Compose(transforms)


class ReferenceHDDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, opt):
        case_name = opt.case_name
        self.dataroot = opt.dataroot
        dataset_name = 'reference_hd'
        self.dataset_name = dataset_name
        self.case_name = case_name
        self.trans = get_image_transform()
        self.one_image_times = 400
        self.images_dir = ospj(opt.dataroot, dataset_name, opt.case_name, 'art')
        sketch_mode = opt.sketch_mode
        if sketch_mode == 'c':
            sketch_dir = 'crisp'
        elif sketch_mode == 's':
            sketch_dir = 'sketch'
        elif sketch_mode == 'l':
            sketch_dir = 'label'
        else:
            sketch_dir = None
            raise ValueError
        self.sketches_dir = ospj(opt.dataroot, dataset_name, opt.case_name, sketch_dir)
        self.images_files = sorted(os.listdir(self.images_dir))
        self.sketches_files = sorted(os.listdir(self.sketches_dir))
        assert len(self.images_files) == len(self.sketches_files)
        self.num_images = len(self.images_files)
        self.size_bound = (512, 512)
        self.length = int(self.num_images * self.one_image_times)

    def __getitem__(self, index):
        img_idx = int(index // self.one_image_times)
        img_path = ospj(self.images_dir, self.images_files[img_idx])
        skt_path = ospj(self.sketches_dir, self.sketches_files[img_idx])
        img = Image.open(img_path).convert('RGB')
        skt = Image.open(skt_path).convert('RGB')
        skt_w, skt_h = skt.size
        img = img.resize((skt_w, skt_h), Image.ANTIALIAS)

        cropped_img, cropped_skt = arbitrary_aspect_ratio(img, skt, self.size_bound)

        cropped_img, cropped_skt = random_flip(cropped_img, cropped_skt)
        # cropped_img_hd, cropped_img, cropped_skt = random_scale_stretch_hd(cropped_img_hd, cropped_img, cropped_skt, self.size_bound)
        # reference_bank = arbitrary_reference_fetch(self.dataroot, self.dataset_name, self.case_name, self.reference_num, self.sample_per_ref)
        # ref_bank = transform_references(reference_bank, skt_size, self.trans)
        return {
            "label": self.trans(cropped_skt), "image": self.trans(cropped_img),
        }

    def __len__(self):
        return self.length
