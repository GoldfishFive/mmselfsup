# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import torch
from mmcls.models import VisionTransformer
import numpy as np
from mmselfsup.registry import MODELS
from ..utils import build_2d_sincos_position_embedding
from ...structures import SelfSupDataSample
import torchvision.transforms as transform

@MODELS.register_module()
class SegMAEVit_VR(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 output_cls_token: bool = True,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 mask_ratio: float = 0.75,
                 fix_mask_ratio: bool = True,
                 max_epochs: int = 300,
                 low_mask_ratio: float = 0.35,
                 high_mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.max_epochs = max_epochs
        self.fix_mask_ratio = fix_mask_ratio
        self.low_mask_ratio = low_mask_ratio
        self.high_mask_ratio = high_mask_ratio

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches ** .5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

    def set_epoch(self, eopch):
        self.cur_epoch = eopch

    def random_masking(
            self,
            x: torch.Tensor,
            mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                masked image, mask and the ids to restore original image.
                  - x_masked (torch.Tensor): masked image.
                  - mask (torch.Tensor): mask used to mask image.
                  - ids_restore (torch.Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_mask_ratio(self):
        if self.fix_mask_ratio:
            return self.mask_ratio
        else:
            low_mask_ratio = self.low_mask_ratio
            high_mask_ratio = self.high_mask_ratio
            total_epoch = self.max_epochs
            cur_epoch_mask_ratio = (high_mask_ratio - low_mask_ratio) * self.cur_epoch / total_epoch + low_mask_ratio

            return cur_epoch_mask_ratio

    def get_image_mask_ratio(self, entropy):
        if entropy < 6:
            return self.high_mask_ratio  # 0.75
        else:
            return self.high_mask_ratio - 0.1 * (entropy - 6) * (self.high_mask_ratio - self.low_mask_ratio) / (
                        0.8 - 0.6)   # high to low


    def seg_var_ratio_masking(self, x: torch.Tensor, data_sample: List[SelfSupDataSample]):
        # print('cur_epoch:', self.cur_epoch)
        mask_ratio = self.get_mask_ratio()

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        len_mask = L - len_keep
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        mask_xsize = W / self.img_size[0]
        mask_ysize = H / self.img_size[1]
        batch_x_masked = []
        batch_mask = []
        batch_ids_restore = []
        batch_ids_keep = []
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        batch_keep_len =  []
        for i in range(N):
            fh_mask = data_sample[i].gt_seg_mask
            # print(fh_mask, fh_mask.shape)
            # parts = data_sample[i].num_of_objects
            entropy = data_sample[i].entropy
            # unique_part = data_sample[i].unique_part
            img_ids_keep = torch.Tensor([]).to(x.device)
            img_ids_mask = torch.Tensor([]).to(x.device)
            img_mask = torch.ones([L], device=x.device)
            # img_ids_restore = torch.Tensor([]).to(x.device)

            # print('image entropy: ', entropy)
            img_mask_ratio = self.get_image_mask_ratio(entropy)
            image_noise = noise[i]

            resize_fh_mask = cv2.resize(np.array(fh_mask, dtype=np.uint8),
                                          dsize=None, fx=mask_xsize, fy=mask_ysize,
                                          interpolation=cv2.INTER_NEAREST)  # dsize=(36,36)
            resize_fh_mask = torch.from_numpy(resize_fh_mask).to(x.device)

            unique_part = torch.unique(resize_fh_mask)
            # for part in image: mask out this ratio. than the whole image mask out around this ratio
            for p in unique_part:
                part_mask = resize_fh_mask == p
                num_of_part = part_mask.sum()
                part_mask_len = int(img_mask_ratio * num_of_part)

                part_noise = image_noise * part_mask.view(-1)

                part_ids_shuffle = torch.argsort(part_noise, dim=0)
                if part_mask_len > 0:
                    part_ids_keep = part_ids_shuffle[-num_of_part:-part_mask_len] # small is keep, large is remove
                    part_ids_mask = part_ids_shuffle[-part_mask_len:]
                else:
                    part_ids_keep = part_ids_shuffle[-num_of_part:]  # small is keep, large is remove
                    part_ids_mask = torch.Tensor([]).to(x.device)

                # part_ids_restore = torch.argsort(part_ids_shuffle, dim=0)[-num_of_part:-part_mask_len] # have problom, whole sorted
                img_ids_keep = torch.cat((img_ids_keep, part_ids_keep), dim=0)
                # img_ids_restore = torch.cat((img_ids_restore, part_ids_restore), dim=0)
                img_ids_mask = torch.cat((img_ids_mask, part_ids_mask))
                # print(img_ids_keep, img_ids_mask)

            img_x = torch.gather(
                x[i], dim=0, index=img_ids_keep.long().unsqueeze(-1).repeat(1, D))
            batch_x_masked.append(img_x)
            img_mask[img_ids_keep.long()] = 0 # 0 is keep,1 is remove
            batch_mask.append(img_mask)
            img_ids_shuffle = torch.cat((img_ids_keep, img_ids_mask))
            img_ids_restore = torch.argsort(img_ids_shuffle)
            # aaaa,bbb = torch.unique(img_ids_shuffle), len(torch.unique(img_ids_shuffle))

            batch_keep_len.append(img_ids_keep.shape[0])
            batch_ids_keep.append(img_ids_keep)
            batch_ids_restore.append(img_ids_restore)

        max_len = max(batch_keep_len)
        # padding zero for equal len  in a batch
        for i in range(N):
            if batch_keep_len[i] < max_len:
                padding_zero = torch.ones([max_len - batch_keep_len[i], D], device=x.device)
                batch_x_masked[i] = torch.cat((batch_x_masked[i], padding_zero))

        batch_x_masked = torch.stack(batch_x_masked)
        batch_mask = torch.vstack(batch_mask)
        batch_ids_restore = torch.vstack(batch_ids_restore)

        return batch_x_masked, batch_mask, batch_ids_restore

    def seg_random_masking(self, x: torch.Tensor, data_sample: List[SelfSupDataSample]):
        # print('cur_epoch:', self.cur_epoch)
        mask_ratio = self.get_mask_ratio()

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        len_mask = L - len_keep
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        mask_xsize = W / self.img_size[0]
        mask_ysize = H / self.img_size[1]
        batch_mask = None
        batch_ids_restore = None
        batch_ids_keep = None

        for i in range(N):
            # fh_mask = data_sample[i].seg_mask
            fh_mask = data_sample[i].gt_seg_mask
            # print(fh_mask, fh_mask.shape)
            parts = data_sample[i].num_of_objects
            ids_keep = None
            mask = None
            restore_ids = None
            part_map = None
            idx = torch.randperm(parts)  # 是把0到parts这些数随机打乱得到的一个数字序列,按这个数字序列依次取parts的mask做掩模
            for p in range(parts):
                id = idx[p].numpy()
                part_mask = fh_mask == id  # 取出对应part的mask
                # print(fh_mask, type(fh_mask))
                # print(part_mask, type(part_mask))
                resize_part_mask = cv2.resize(np.array(part_mask, dtype=np.uint8),
                                              dsize=None, fx=mask_xsize, fy=mask_ysize,
                                              interpolation=cv2.INTER_NEAREST)  # dsize=(36,36)
                if p == 0:
                    part_map = resize_part_mask
                else:
                    part_map += resize_part_mask
                num_of_part = part_map.sum()
                flatten_part_map = part_map.flatten()
                mask_ids = np.argsort(flatten_part_map, axis=None)  # sort from small to large
                restore_ids = np.argsort(mask_ids)
                if num_of_part < len_mask:
                    continue
                elif num_of_part >= len_mask:
                    ids_keep = mask_ids[:len_keep]
                    flatten_part_map[mask_ids[:len_keep]] = 0
                    mask = flatten_part_map
                    break

            ids_keep = ids_keep
            mask = mask
            ids_restore = restore_ids
            if i == 0:
                batch_ids_keep = ids_keep
                batch_mask = mask
                batch_ids_restore = ids_restore
            else:
                batch_ids_keep = np.vstack([batch_ids_keep, ids_keep])
                batch_mask = np.vstack([batch_mask, mask])
                batch_ids_restore = np.vstack([batch_ids_restore, ids_restore])

        batch_ids_keep = torch.from_numpy(batch_ids_keep).to(x.device)
        batch_mask = torch.from_numpy(batch_mask).to(x.device)
        batch_ids_restore = torch.from_numpy(batch_ids_restore).to(x.device)
        batch_x_masked = torch.gather(
            x, dim=1, index=batch_ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return batch_x_masked, batch_mask, batch_ids_restore

    def forward(
            self, x: torch.Tensor,
            data_sample: List[SelfSupDataSample]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        This function generates mask and masks some patches randomly and get
        the hidden features for visible patches.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

            Hidden features, mask and the ids to restore original image.

                - x (torch.Tensor): hidden features, which is of shape
                  B x (L * mask_ratio) x C.
                - mask (torch.Tensor): mask used to mask image.
                - ids_restore (torch.Tensor): ids to restore original image.
        """
        B = x.shape[0]

        x = self.patch_embed(x)[0]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        # x, mask, ids_restore = self.seg_random_masking(x, data_sample)
        x, mask, ids_restore = self.seg_var_ratio_masking(x, data_sample)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for _, layer in enumerate(self.layers):
            x = layer(x)
        # Use final norm
        x = self.norm1(x)

        return (x, mask, ids_restore)
