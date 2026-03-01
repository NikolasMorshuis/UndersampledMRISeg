import numpy as np
import torch

from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import SimpleDataloader
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from torch.utils.data import DataLoader

from nnunet.training.network_training.nnUNet_variants.paper_methods.nnUNetTrainer_RecSeg import (
    nnUNetTrainer_RecSeg,
)


from nnunet.training.loss_functions.RecSeg_baseline_loss import RecSeg_baseline_loss
from nnunet.network_architecture.One_Encoder_Two_Decoders import (
    One_Encoder_Two_Decoders,
)
from nnunet.network_architecture.generic_modular_UNet import get_default_network_config


class nnUNetTrainer_OneEncTwoDec(nnUNetTrainer_RecSeg):
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans_seg)
            self.process_plans_reg(self.plans_reg)
            self.batch_size = 2
            self.batch_size_reg = 2

            self.deterministic_val = False
            self.unpack_data = True  # already unpacked for now

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True]
                + [
                    True if i < net_numpool - 1 else False
                    for i in range(1, net_numpool)
                ]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            if "skm" in self.task_name:
                self.loss = RecSeg_baseline_loss(
                    batch_dice=self.batch_dice, seg_loss_weight=5.0, rec_loss_weight=1.0
                )
            else:
                self.loss = RecSeg_baseline_loss(
                    batch_dice=self.batch_dice, seg_loss_weight=1.0, rec_loss_weight=1.0
                )
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.deterministic_val:
                    Dataset_Deterministic_Val = SimpleDataloader
                    val_dataset = Dataset_Deterministic_Val(
                        self.dataset_val, self.deep_supervision_scales, self.classes
                    )
                    # item = val_dataset.__getitem__(0)
                    # self.dl_val = SimpleDataloader(self.dataset_val, 1)
                    self.num_val_batches_per_epoch = val_dataset.__len__()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen, self.val_gen = get_no_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    params=self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                )

                if self.deterministic_val:
                    # val_aug_params = self.data_aug_params
                    # val_aug_params["num_cached_per_thread"] = 1
                    # val_aug_params["num_threads"] = 1
                    # __, self.val_gen = get_no_augmentation(
                    #    self.dl_tr, self.dl_val,
                    #    val_aug_params,
                    #    deep_supervision_scales=self.deep_supervision_scales,
                    #    pin_memory=self.pin_memory
                    # )
                    __ = None
                    self.val_gen = DataLoader(
                        val_dataset, batch_size=1, shuffle=False, num_workers=6
                    )
                    it = iter(self.val_gen)
                    a = next(it)

                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass
            self.initialize_network()  # segmentation part
            self.initialize_optimizer_and_scheduler()
            # self.load_checkpoints_wrapped_model()

            # plan corrections:
            self.process_plans(self.plans_reg)
            self.num_classes = 7
            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        base_num_features_seg = 16  # Also half the number of the typical parameters
        net_num_pool_op_kernel_sizes_here = (
            list([[1, 1, 1]]) + self.net_num_pool_op_kernel_sizes
        )

        self.network = One_Encoder_Two_Decoders(
            input_channels=1,
            base_num_features=base_num_features_seg,
            num_classes=self.num_classes,
            num_blocks_per_stage_decoder=2,
            num_blocks_per_stage_encoder=2,
            feat_map_mul_on_downscale=2,
            pool_op_kernel_sizes=net_num_pool_op_kernel_sizes_here,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            props=get_default_network_config(3, dropout_p=None),
            upscale_logits=False,
            max_features=512,
            initializer=InitWeights_He(1e-2),
            deep_supervision=True,
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
