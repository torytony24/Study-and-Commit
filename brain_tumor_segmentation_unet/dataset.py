from torch.utils.data import Dataset
import nibabel as nib  
import os
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, root_dir, cases_list, transform=None, range_width=20, step=2):
        """
        root_dir: base folder containing MICCAI_BraTS2020_TrainingData
        cases_list: list of case folder names, e.g. ['BraTS20_Training_001', ...]
        transform: optional transform to apply to image and mask
        range_width: how far from center slice to include (e.g., 20 → use center±20)
        step: interval between slices (e.g., 2 → every 2nd slice)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.slice_info_list = []

        for case in cases_list:
            case_path = os.path.join(root_dir, case)
            t1 = nib.load(os.path.join(case_path, f"BraTS20_Training_{case}_t1.nii")).get_fdata()
            num_slices = t1.shape[2]
            center_slice = num_slices // 2

            start = max(0, center_slice - range_width)
            end = min(num_slices, center_slice + range_width + 1)

            for slice_idx in range(start, end, step):
                self.slice_info_list.append((case, slice_idx))


    def __len__(self):
        return len(self.slice_info_list)

    def __getitem__(self, idx):
        case, slice_idx = self.slice_info_list[idx]
        case_path = os.path.join(self.root_dir, case)

        t1 = nib.load(os.path.join(case_path, f"BraTS20_Training_{case}_t1.nii")).get_fdata()
        seg = nib.load(os.path.join(case_path, f"BraTS20_Training_{case}_seg.nii")).get_fdata()

        image_slice = t1[:, :, slice_idx]
        mask_slice = (seg[:, :, slice_idx] > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

class ValidDataset(Dataset):
    def __init__(self, root_dir, cases_list, transform=None):
        """
        root_dir: base folder containing MICCAI_BraTS2020_TrainingData
        cases_list: list of case folder names, e.g. ['BraTS20_Training_001', ...]
        transform: optional transform to apply to image and mask
        """
        self.root_dir = root_dir
        self.cases_list = cases_list
        self.transform = transform

    def __len__(self):
        return len(self.cases_list)

    def __getitem__(self, idx):
        case = self.cases_list[idx]
        case_path = os.path.join(self.root_dir, case)

        t1 = nib.load(os.path.join(case_path, f"BraTS20_Training_{case}_t1.nii")).get_fdata()
        seg = nib.load(os.path.join(case_path, f"BraTS20_Training_{case}_seg.nii")).get_fdata()

        # Here we take the middle slice of the volume for 2D training
        slice_idx = t1.shape[2] // 2
        image_slice = t1[:, :, slice_idx]
        mask_slice = (seg[:, :, slice_idx] > 0).astype(np.float32)

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()    # (1, H, W)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

def to_three_digit_string(number):
    return "{:03d}".format(number)

train_root = "C:/Users/Sanghwa/data/train"
train_list = ["{:03d}".format(i) for i in range(1, 301)]
train_dataset = TrainDataset(train_root, train_list)

valid_root = "C:/Users/Sanghwa/data/val"
valid_list = ["{:03d}".format(i) for i in range(301, 351)]
valid_dataset = ValidDataset(valid_root, valid_list)