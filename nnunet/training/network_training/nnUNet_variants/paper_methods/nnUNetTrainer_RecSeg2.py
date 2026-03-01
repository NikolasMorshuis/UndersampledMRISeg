import numpy as np
import torch

# from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import SimpleDataloader
from nnunet.training.data_augmentation.data_augmentation_mirror_only import (
    get_mirror_augmentation,
)
from torch.utils.data import DataLoader
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from time import time
import torch.backends.cudnn as cudnn
from tqdm import trange
from nnunet.training.network_training.nnUNet_variants.paper_methods.nnUNetTrainer_RegSeg2 import (
    nnUNetTrainer_RegSeg2,
)

from _warnings import warn

from nnunet.training.loss_functions.RecSeg_baseline_loss import RecSeg_baseline_loss


class model_wrapper(SegmentationNetwork):
    # batchnorm in order to simulate the normalization preprocessing step and to make it differentiable
    def __init__(self, network_reg, network_seg):
        super(model_wrapper, self).__init__()
        self.network_reg = network_reg
        self.network_seg = network_seg
        self.conv_op = nn.Conv3d
        self.do_ds = True
        self.num_classes = 7
        self.inference_apply_nonlin = softmax_helper

    def forward(self, x):
        rec = self.network_reg(x)
        # both = torch.cat((rec, x), 1)
        seg = self.network_seg(rec)
        if self.do_ds:
            return rec, seg
        else:
            return rec[0], seg[0]


class nnUNetTrainer_RecSeg2(nnUNetTrainer_RegSeg2):
    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        super(nnUNetTrainer_RegSeg2, self).__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )
        self.max_num_epochs = 1000
        self.initial_lr = 1e-3  # factor 10 less then the V2 network. We just want to nudge the network here
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.fold = fold

        # self.num_batches_per_epoch = 2  #debug purposes

        # We need to let the seg network use 2 channels again
        self.task_name = plans_file.split("/")[-2]

        self.task_id = self.task_name.split("_")[0][-3:]
        self.task_id = int(self.task_id)
        self.task_id_pretrained = (
            self.task_id
        )  # here we want to have the standard seg method
        self.task_name_pretrained = convert_id_to_task_name(self.task_id_pretrained)

        self.plans_file_seg = os.path.join(
            "/mnt/qb/work/baumgartner/jmorshuis45/data/k2s_data/nnUNet_preprocessed",
            self.task_name_pretrained,
            "nnUNetPlansv2.1_plans_3D.pkl",
        )
        # self.plans_file_seg = "/mnt/qb/work/baumgartner/jmorshuis45/data/k2s_data/nnUNet_preprocessed/Task722_acc_32_unders_and_fullys/nnUNetPlansv2.1_plans_3D.pkl"
        self.plans_file_reg = self.plans_file_seg
        self.load_plans_file_reg()
        self.load_plans_file_seg()
        self.plans_file = self.plans_file_reg
        # self.num_batches_per_epoch = 5 # debug purposes
        # weights_classes = torch.Tensor([0.366635, 136.706224, 398.612625, 667.377409, 8.573840, 12.130584, 89.331420])/100

        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, weights_classes=weights_classes)
        self.start_weightchange = 99999  # never
        self.all_weighted_val_eval_metrics = []
        if "skm" in self.task_name:
            self.loss = RecSeg_baseline_loss(
                batch_dice=self.batch_dice, seg_loss_weight=1.0, rec_loss_weight=1.0
            )
        else:
            self.loss = RecSeg_baseline_loss(
                batch_dice=self.batch_dice, seg_loss_weight=1.0, rec_loss_weight=1.0
            )

    def setup_DA_params(self):
        super().setup_DA_params()
        # Mirroring should not hurt the regression task.
        # We also do not expect much longer training times
        self.data_aug_params["do_mirror"] = True  # change this for k2s True
        if "skm" in self.task_name:
            self.data_aug_params["mirror_axes"] = (
                0,
            )  # I think it is 2 for k2s and it should be 0 for skm
        else:
            self.data_aug_params["mirror_axes"] = (2,)

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
                    batch_dice=self.batch_dice, seg_loss_weight=1.0, rec_loss_weight=1.0
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

                # use mirror for k2s but not for skm
                self.tr_gen, self.val_gen = get_mirror_augmentation(
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
            self.initialize_network_seg()  # segmentation part
            self.initialize_network_reg()  # regression part
            self.network = model_wrapper(
                self.network_reg, self.network
            )  # combines seg and reg
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

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = None

    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        input_img = data[:, 1:2]  # select the undersampled data as input
        target_rec = data[:, 0:1]  # select the fully sampled data as target
        del data

        if torch.cuda.is_available():
            input_img = to_cuda(input_img)
            target_rec = to_cuda(target_rec)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output_rec, output_seg = self.network(input_img)
                del input_img
                l = self.loss(output_rec, output_seg, target_rec, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output_rec, output_seg = self.network(input)
            del input
            l = self.loss(output_rec, output_seg, target_rec, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output_seg, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0].unsqueeze(0)
        output = output[0].unsqueeze(0)

        # one-hot encode the output and the target
        # output = to_one_hot(output, num_classes=7)

        return super().run_online_evaluation(output, target)

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!"
            )

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn(
                "torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                "If you want deterministic then set benchmark=False"
            )

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description(
                            "Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs)
                        )

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file(
                    "validation loss: %.4f" % self.all_val_losses[-1]
                )

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file(
                        "validation loss (train=True): %.4f"
                        % self.all_val_losses_tr_mode[-1]
                    )

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file(
                "This epoch took %f s\n" % (epoch_end_time - epoch_start_time)
            )

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint:
            self.save_checkpoint(
                join(self.output_folder, "model_final_checkpoint.model")
            )
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def initialize_network_seg(self):
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

        self.network = Generic_UNet(
            1,
            base_num_features_seg,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            True,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [
            i
            for i in [
                2 * i / (2 * i + j + k)
                for i, j, k in zip(
                    self.online_eval_tp, self.online_eval_fp, self.online_eval_fn
                )
            ]
            if not np.isnan(i)
        ]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file(
            "Average global foreground Dice:",
            [np.round(i, 4) for i in global_dc_per_class],
        )
        self.print_to_log_file(
            "(interpret this as an estimate for the Dice of the different classes. This is not "
            "exact.)"
        )

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
