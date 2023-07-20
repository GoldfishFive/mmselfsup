# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
from mmcls.datasets import CustomDataset
from mmengine.fileio import join_path

from mmselfsup.registry import DATASETS
from mmengine.fileio import list_from_file

@DATASETS.register_module()
class SegMAEImageList(CustomDataset):
    """The dataset implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.

    An annotation file should be provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       1. If data is labeled, the annotation file (the first column is the image
       path and the second column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

        2. If data is unlabeled, the annotation file is: ::

            folder_1/xxx.png
            folder_1/xxy.png
            folder_2/123.png
            folder_2/nsdf3.png
            ...

    Args:
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (str | dict): Prefix for training data. Defaults
            to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 ann_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 **kwargs) -> None:
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            lazy_init=True,
            **kwargs)
        self.ann_root = ann_root
        print('self.ann_root', self.ann_root)

    def load_data_list(self) -> List[dict]:
        """Rewrite load_data_list() function for supporting annotation files
        with unlabeled data.

        Returns:
            List[dict]: A list of data information.
        """

        samples = self._find_samples()

        data_list = []
        # print(samples)
        for filename, gt_label in samples:
            img_path = join_path(self.img_prefix, filename)
            fh_seg_path = join_path(self.ann_root, filename.replace('JPEG','npy'))
            # print('fh_seg_path', fh_seg_path)
            info = {'img_path': img_path, 'gt_label': int(gt_label), 'fh_seg': fh_seg_path}
            data_list.append(info)
        return data_list
