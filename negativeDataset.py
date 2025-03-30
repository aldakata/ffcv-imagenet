import numpy as np
import torch
from torchvision.datasets import ImageFolder
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json

def _process_directory(directory_charge: str, class_name: str, root_dir: str, is_positive: bool, class_to_idx: Dict[str, int], class_sample_counts: int) -> List[Tuple[str, int, bool]]:
    results = []
    target_idx = class_to_idx[class_name]
    class_dir = os.path.join(directory_charge, class_name)
    
    if not os.path.isdir(class_dir):
        return results
        
    for root, _, fnames in sorted(os.walk(class_dir)):
        if not is_positive and class_sample_counts != -1:
            fnames = np.random.choice(
                fnames, 
                min(class_sample_counts, len(fnames)), 
                replace=False
            ).tolist()
        
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            item = (path, target_idx, is_positive)
            results.append(item)
    return results
def _process_directory_neg(directory_charge: str, class_name: str, root_dir: str, is_positive: bool, class_to_idx: Dict[str, int], class_sample_counts: int, sampling_method: str) -> List[Tuple[str, int, bool]]:
    results = []
    target_idx = class_to_idx[class_name]
    class_dir = os.path.join(directory_charge, class_name)
    
    if not os.path.isdir(class_dir):
        return results
        
    for root, _, fnames in sorted(os.walk(class_dir)):
        if not is_positive and class_sample_counts != -1:
            if sampling_method == "random":
                fnames = np.random.choice(
                    fnames, 
                    min(class_sample_counts, len(fnames)), 
                    replace=False
                ).tolist()
            elif sampling_method == 'sorted':
                fnames = sorted(fnames)[:min(class_sample_counts, len(fnames)-1)]   
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            item = (path, target_idx, is_positive)
            results.append(item)
    return results

class NegativeDataset(ImageFolder):
    def __init__(self, root, root_pos, root_neg, class_sample_counts = -1, transform=None, target_transform=None, num_workers=8, negative_samples_file=None, sampling_method="random",**kwargs):
        self.root_pos = root_pos
        self.root_neg = root_neg
        self.root = root
        self.class_sample_counts = class_sample_counts
        self.num_workers = num_workers
        self.negative_samples_file = negative_samples_file
        self.sampling_method=sampling_method
        # If negative samples file is provided, use it to load samples
        if negative_samples_file and os.path.exists(negative_samples_file):
            self.make_dataset = self.load_samples_from_file_concurrent
        else:
            self.make_dataset = self.make_dataset_concurrent
            
        super(NegativeDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.charges = [s[2] for s in self.samples]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(self.root_pos) if entry.is_dir()) # using self.root_pos because maybe some classes don't have negative samples 
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.root_pos}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        **kwargs
        ) -> List[Tuple[str, int, bool]]:
        """
            Generates a list of samples of a form (path_to_sample, class).
            See :class:`DatasetFolder` for details.
            Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
            by default.

            Modified to accept the number of negative samples for each class.
        """
        images = []

        # Negative samples
        directory = os.path.expanduser(self.root_neg) # directory is the folder containing root_pos
        for directory_charge, _, _ in sorted(os.walk(directory)):
            for target in sorted(class_to_idx.keys()):
                d = os.path.join(directory_charge, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    if self.class_sample_counts!=-1:
                        fnames = np.random.choice(fnames, min(self.class_sample_counts, len(fnames)), replace=False).tolist()
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = path, class_to_idx[target], False
                        images.append(item)

        # Positive samples
        directory = os.path.expanduser(self.root_pos) # directory is the folder containing root_pos
        for directory_charge, _, _ in sorted(os.walk(directory)):
            for target in sorted(class_to_idx.keys()):
                d = os.path.join(directory_charge, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = path, class_to_idx[target], True
                        images.append(item)
        return images

    def make_dataset_concurrent(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        num_workers: int = 8,
        **kwargs
        ) -> List[Tuple[str, int, bool]]:
        """
        Concurrent version of make_dataset that uses multiple processes to speed up file listing.
        """
        images = []
        classes = sorted(class_to_idx.keys())

        # Process negative samples
        directory_neg = os.path.expanduser(self.root_neg)
        directory_charges_neg = [d[0] for d in sorted(os.walk(directory_neg))][0:1]  # Get only the first level
        
        for directory_charge in sorted(directory_charges_neg):
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Use the global _process_directory function directly
                neg_process_fn = partial(_process_directory_neg, 
                                      directory_charge, 
                                      root_dir=self.root_neg, 
                                      is_positive=False, 
                                      class_to_idx=class_to_idx,
                                      class_sample_counts=self.class_sample_counts,
                                      sampling_method=self.sampling_method
                                      )
                neg_results = list(executor.map(neg_process_fn, classes))
                for result in neg_results:
                    images.extend(result)

        # Process positive samples
        directory_pos = os.path.expanduser(self.root_pos)
        directory_charges_pos = [d[0] for d in sorted(os.walk(directory_pos))][0:1]  # Get only the first level
        
        for directory_charge in sorted(directory_charges_pos):
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                pos_process_fn = partial(_process_directory, 
                                      directory_charge, 
                                      root_dir=self.root_pos, 
                                      is_positive=True,
                                      class_to_idx=class_to_idx,
                                      class_sample_counts=self.class_sample_counts)
                pos_results = list(executor.map(pos_process_fn, classes))
                for result in pos_results:
                    images.extend(result)
        return images

    def save_negative_samples_to_file(self, output_file: str):
        """
        Save all negative samples (path and target) to a JSON file
        """
        negative_samples = [(path, target) for path, target, is_positive in self.samples if not is_positive]
        with open(output_file, 'w') as f:
            json.dump(negative_samples, f)
        
        print(f"Saved {len(negative_samples)} negative samples to {output_file}")

    def load_samples_from_file_concurrent(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        num_workers: int = 8,
        **kwargs
        ) -> List[Tuple[str, int, bool]]:
        """
        Concurrent version of make_dataset that uses multiple processes to speed up file listing.
        """
        images = []
        classes = sorted(class_to_idx.keys())

        with open(self.negative_samples_file, 'r') as f:
            negative_samples = json.load(f)

        target_to_samples = {}
        for path, target in negative_samples:
            if target not in target_to_samples:
                target_to_samples[target] = []
            target_to_samples[target].append(path)

        # Adjust negative samples based on class_sample_counts
        for target, samples in target_to_samples.items():
            if self.class_sample_counts != -1:
                if len(samples) > self.class_sample_counts:
                    # Subsample if we have too many
                    sample_idcs = np.arange(0, len(samples))
                    selected_idcs = np.random.choice(
                        sample_idcs, 
                        self.class_sample_counts, 
                        replace=False
                    ).tolist()
                    selected_samples = np.asarray(samples)[selected_idcs]
                    target_to_samples[target] = selected_samples
                elif len(samples) < self.class_sample_counts:
                    # Prepare for finding additional samples
                    class_name = [k for k, v in class_to_idx.items() if v == target][0]
                    existing_paths = set(samples)  # Already selected paths
                    class_dir = os.path.join(self.root_neg, class_name)
                    samples_needed = self.class_sample_counts - len(samples)

                    # Find additional samples
                    additional_samples = self._find_additional_samples(class_dir, target, existing_paths, samples_needed)
                    target_to_samples[target].extend(additional_samples)

       
        images = [(path, target, False) for target, paths in target_to_samples.items() for path in paths]

        # Process positive samples
        directory_pos = os.path.expanduser(self.root_pos)
        directory_charges_pos = [d[0] for d in sorted(os.walk(directory_pos))][0:1]  # Get only the first level
        
        for directory_charge in sorted(directory_charges_pos):
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                pos_process_fn = partial(_process_directory, 
                                      directory_charge, 
                                      root_dir=self.root_pos, 
                                      is_positive=True,
                                      class_to_idx=class_to_idx,
                                      class_sample_counts=self.class_sample_counts)
                pos_results = list(executor.map(pos_process_fn, classes))
                for result in pos_results:
                    images.extend(result)
        return images

    def _find_additional_samples(self, class_dir: str, target: int, existing_paths: set, samples_needed: int) -> List[Tuple[str, int]]:
        """Helper function to find additional samples that are not already selected."""
        additional_samples = []
        
        if not os.path.isdir(class_dir):
            return additional_samples

        all_paths = []
        for root, _, fnames in sorted(os.walk(class_dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if path not in existing_paths:
                    all_paths.append(path)

        if all_paths:
            num_to_select = min(samples_needed, len(all_paths))
            selected_paths = np.random.choice(all_paths, num_to_select, replace=False)
            additional_samples.extend(str(path) for path in selected_paths)
        
        return additional_samples

    def __getitem__(self, index: int) -> Tuple[Any, int, bool, int]:
        path, target, charge = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, charge

class TargetTransform(Callable):
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, x):
        return self.mapping[x]