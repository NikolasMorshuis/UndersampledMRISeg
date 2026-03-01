from collections import OrderedDict

import numpy as np
import torch

# from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import (
    DataLoader3D,
    DataLoader2D,
    SimpleDataloader,
)
from nnunet.training.data_augmentation.data_augmentation_mirror_only import (
    get_mirror_augmentation,
)
from torch.utils.data import DataLoader
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
import matplotlib
from torch.optim.lr_scheduler import _LRScheduler

from time import time
import matplotlib.pyplot as plt
import sys
import torch.backends.cudnn as cudnn
from tqdm import trange
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from _warnings import warn


def weight_scheduler(epoch, start_epoch, max_epoch, final_weights):
    """We assume we start at equal weight and want to get to final_weight
    We use a linear scheduler"""
    init_weights = np.ones_like(final_weights)
    if epoch > start_epoch:
        stepsize = (final_weights - init_weights) / (max_epoch - start_epoch)
        current_weights = init_weights + (epoch - start_epoch) * stepsize
    else:
        current_weights = init_weights
    return current_weights


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
        both = torch.cat((rec, x), 1)
        if self.do_ds:
            return self.network_seg(both)
        else:
            return self.network_seg(both)[0]


class nnUNetTrainer_RegSeg2(nnUNetTrainerV2):
    # We just take the standard nnUNetTrainerV2. Only thing we have to adjust is the loading of the NNs and the loading
    # of the pretrained weights. Think some hardcoding is ok for the challenge
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
        super().__init__(
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
        self.max_num_epochs = 500
        self.initial_lr = 1e-3  # factor 10 less then the V2 network. We just want to nudge the network here
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.fold = fold
        self.task_name = plans_file.split("/")[-2]

        self.task_id = self.task_name.split("_")[0][-3:]
        self.task_id = int(self.task_id)
        self.task_id_pretrained = self.task_id + 1
        self.task_name_pretrained = convert_id_to_task_name(self.task_id_pretrained)

        self.plans_file_seg = os.path.join(
            os.environ["nnUNet_preprocessed"],
            self.task_name_pretrained,
            "nnUNetPlansv2.1_plans_3D.pkl",
        )
        print("plans_file_seg:", self.plans_file_seg)
        self.plans_file_reg = self.plans_file_seg
        self.load_plans_file_reg()
        self.load_plans_file_seg()
        self.plans_file = self.plans_file_reg
        # self.num_batches_per_epoch = 5 # debug purposes
        # weights_classes = torch.Tensor([0.366635, 136.706224, 398.612625, 667.377409, 8.573840, 12.130584, 89.331420])/100

        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, weights_classes=weights_classes)
        self.start_weightchange = 99999  # never
        self.all_weighted_val_eval_metrics = []

    def load_checkpoints_wrapped_model(self):

        path_reg = os.path.join(
            os.environ["RESULTS_FOLDER"],
            "nnUNet",
            "3d_fullres",
            self.task_name_pretrained,
            "nnUNetTrainer_Reg2__nnUNetPlansv2.1",
            "fold_{}".format(self.fold),
            "model_final_checkpoint.model",
        )  #'model_final_checkpoint.model')
        checkpoint_reg = torch.load(path_reg)
        self.network.network_reg.load_state_dict(checkpoint_reg["state_dict"])

        path_seg = os.path.join(
            os.environ["RESULTS_FOLDER"],
            "nnUNet",
            "3d_fullres",
            self.task_name_pretrained,
            "nnUNetTrainerV2_Seg1__nnUNetPlansv2.1",
            "fold_{}".format(self.fold),
            "model_final_checkpoint.model",
        )
        checkpoint_seg = torch.load(path_seg)
        checkpoint_seg2 = torch.load(path_seg)
        self.network.network_seg.load_state_dict(checkpoint_seg["state_dict"])
        optim_checkpoint_new = checkpoint_seg2["optimizer_state_dict"].copy()
        optim_checkpoint_new["param_groups"][0]["params"] = np.arange(
            len(checkpoint_reg["optimizer_state_dict"]["param_groups"][0]["params"])
            + len(checkpoint_reg["optimizer_state_dict"]["param_groups"][0]["params"])
        )

        for i in checkpoint_reg["optimizer_state_dict"]["state"].keys():
            optim_checkpoint_new["state"][i] = checkpoint_reg["optimizer_state_dict"][
                "state"
            ][i]
        seg_keys = checkpoint_seg["optimizer_state_dict"]["state"].keys()
        for j in seg_keys:
            optim_checkpoint_new["state"][j + i + 1] = checkpoint_seg[
                "optimizer_state_dict"
            ]["state"][j]

        self.optimizer.load_state_dict(optim_checkpoint_new)

        print("checkpoint loaded")

    def load_plans_file_reg(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans_reg = load_pickle(self.plans_file_reg)

    def load_plans_file_seg(self):
        self.plans_seg = load_pickle(self.plans_file_seg)

    def setup_DA_params(self):
        super().setup_DA_params()
        # important because we need to know in validation and inference that we did not mirror in training
        self.data_aug_params["do_mirror"] = True
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
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
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
            self.load_checkpoints_wrapped_model()

            # plan corrections:
            self.process_plans(self.plans_reg)
            self.num_classes = 7
            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(
                self.dataset_tr,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset_tr,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                transpose=self.plans.get("transpose_forward"),
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
            dl_val = DataLoader2D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                transpose=self.plans.get("transpose_forward"),
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
        return dl_tr, dl_val

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(
            self.lr_scheduler, "state_dict"
        ):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            "epoch": self.epoch + 1,
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "lr_scheduler_state_dict": lr_sched_state_dct,
            "plot_stuff": (
                self.all_tr_losses,
                self.all_val_losses,
                self.all_val_losses_tr_mode,
                self.all_val_eval_metrics,
                self.all_weighted_val_eval_metrics,
            ),
            "best_stuff": (
                self.best_epoch_based_on_MA_tr_loss,
                self.best_MA_tr_loss_for_patience,
                self.best_val_eval_criterion_MA,
            ),
        }
        if self.amp_grad_scaler is not None:
            save_this["amp_grad_scaler"] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

        info = OrderedDict()
        info["init"] = self.init_args
        info["name"] = self.__class__.__name__
        info["class"] = str(self.__class__)
        info["plans"] = self.plans

        write_pickle(info, fname + ".pkl")

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint["state_dict"].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if "amp_grad_scaler" in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint["amp_grad_scaler"])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint["epoch"]
        if train:
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if (
                self.lr_scheduler is not None
                and hasattr(self.lr_scheduler, "load_state_dict")
                and checkpoint["lr_scheduler_state_dict"] is not None
            ):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        (
            self.all_tr_losses,
            self.all_val_losses,
            self.all_val_losses_tr_mode,
            self.all_val_eval_metrics,
            self.all_weighted_val_eval_metrics,
        ) = checkpoint["plot_stuff"]

        # load best loss (if present)
        if "best_stuff" in checkpoint.keys():
            (
                self.best_epoch_based_on_MA_tr_loss,
                self.best_MA_tr_loss_for_patience,
                self.best_val_eval_criterion_MA,
            ) = checkpoint["best_stuff"]

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file(
                "WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                "due to an old bug and should only appear when you are loading old models. New "
                "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)"
            )
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[: self.epoch]
            self.all_val_losses = self.all_val_losses[: self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[: self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[: self.epoch]

        self._maybe_init_amp()

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {"weight": "normal", "size": 18}

            matplotlib.rc("font", **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color="b", ls="-", label="loss_tr")

            ax.plot(
                x_values,
                self.all_val_losses,
                color="r",
                ls="-",
                label="loss_val, train=False",
            )

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(
                    x_values,
                    self.all_val_losses_tr_mode,
                    color="g",
                    ls="-",
                    label="loss_val, train=True",
                )
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(
                    x_values,
                    self.all_val_eval_metrics,
                    color="g",
                    ls="--",
                    label="evaluation metric",
                )
            if len(self.all_weighted_val_eval_metrics) == len(x_values):
                ax2.plot(
                    x_values,
                    self.all_weighted_val_eval_metrics,
                    color="k",
                    ls="-.",
                    label="weighted evaluation metric",
                )

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        We need to look for the weighted eval metrics all_weighted_eval_metrics, because this is what we need to optimize for the challenge
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_weighted_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = -self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_weighted_val_eval_metrics[-1]
        else:
            if len(self.all_weighted_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = (
                    self.val_eval_criterion_alpha * self.val_eval_criterion_MA
                    - (1 - self.val_eval_criterion_alpha) * self.all_val_losses[-1]
                )
            else:
                self.val_eval_criterion_MA = (
                    self.val_eval_criterion_alpha * self.val_eval_criterion_MA
                    + (1 - self.val_eval_criterion_alpha)
                    * self.all_weighted_val_eval_metrics[-1]
                )

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, 0.9
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )

    def finish_online_evaluation(self, weights_classes=None):
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
        if weights_classes is not None:
            weighted_dice = np.sum((weights_classes[1:] * global_dc_per_class)) / (
                np.sum(weights_classes)
            )
            self.all_weighted_val_eval_metrics.append(weighted_dice)
        else:
            self.all_weighted_val_eval_metrics = self.all_val_eval_metrics

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

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!"
            )

        _ = self.tr_gen.next()
        if self.deterministic_val:
            pass
        else:
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
            if self.epoch == 0:
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        if self.deterministic_val:
                            it = iter(self.val_gen)
                            l = self.run_iteration(it, False, True)
                        else:
                            l = self.run_iteration(self.val_gen, False, True)
                        val_losses.append(l)
                    self.finish_online_evaluation(weights_classes=None)
                    self.print_to_log_file(
                        "validation loss start (dice): %.4f"
                        % self.all_val_eval_metrics[0]
                    )
                    self.all_val_eval_metrics = []
                    self.all_weighted_val_eval_metrics = []

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
                    if self.deterministic_val:
                        it = iter(self.val_gen)
                        l = self.run_iteration(it, False, True)
                    else:
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
            if self.epoch >= self.start_weightchange:
                # self.weights_classes_current = weight_scheduler(self.epoch, self.start_weightchange, self.max_num_epochs, self.weights_classes)
                self.loss = DC_and_CE_loss(
                    {"batch_dice": self.batch_dice, "smooth": 1e-5, "do_bg": False}, {}
                )
                self.loss = MultipleOutputLoss2(self.loss)

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

    def process_plans_reg(self, plans):
        if self.stage is None:
            assert len(list(plans["plans_per_stage"].keys())) == 1, (
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the "
                "case. Please specify which stage of the cascade must be trained"
            )
            self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans_reg = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size_reg = stage_plans["batch_size"]
        self.net_pool_per_axis_reg = stage_plans["num_pool_per_axis"]
        self.patch_size_reg = np.array(stage_plans["patch_size"]).astype(int)
        self.do_dummy_2D_aug_reg = stage_plans["do_dummy_2D_data_aug"]

        if "pool_op_kernel_sizes" not in stage_plans.keys():
            assert "num_pool_per_axis" in stage_plans.keys()
            self.print_to_log_file(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it..."
            )
            self.net_num_pool_op_kernel_sizes_reg = []
            for i in range(max(self.net_pool_per_axis_reg)):
                curr = []
                for j in self.net_pool_per_axis_reg:
                    if (max(self.net_pool_per_axis_reg) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes_reg.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes_reg = stage_plans["pool_op_kernel_sizes"]

        if "conv_kernel_sizes" not in stage_plans.keys():
            self.print_to_log_file(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it..."
            )
            self.net_conv_kernel_sizes_reg = [[3] * len(self.net_pool_per_axis_reg)] * (
                max(self.net_pool_per_axis_reg) + 1
            )
        else:
            self.net_conv_kernel_sizes_reg = stage_plans["conv_kernel_sizes"]

        self.pad_all_sides_reg = None  # self.patch_size
        self.intensity_properties_reg = plans["dataset_properties"][
            "intensityproperties"
        ]
        self.normalization_schemes_reg = plans["normalization_schemes"]
        self.base_num_features_reg = plans["base_num_features"]
        self.num_input_channels_reg = plans["num_modalities"]
        self.num_classes_reg = (
            plans["num_classes"] + 1
        )  # background is no longer in num_classes
        self.classes_reg = plans["all_classes"]
        self.use_mask_for_norm_reg = plans["use_mask_for_norm"]
        self.only_keep_largest_connected_component_reg = plans[
            "keep_only_largest_region"
        ]
        self.min_region_size_per_class_reg = plans["min_region_size_per_class"]
        self.min_size_per_class_reg = None  # DONT USE THIS. plans['min_size_per_class']

        if (
            plans.get("transpose_forward") is None
            or plans.get("transpose_backward") is None
        ):
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            plans["transpose_forward"] = [0, 1, 2]
            plans["transpose_backward"] = [0, 1, 2]
        self.transpose_forward_reg = plans["transpose_forward"]
        self.transpose_backward_reg = plans["transpose_backward"]

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

        if (
            "conv_per_stage" in plans.keys()
        ):  # this ha sbeen added to the plans only recently
            self.conv_per_stage_reg = plans["conv_per_stage"]
        else:
            self.conv_per_stage_reg = 2

    def initialize_network_reg(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        softmax_helper = lambda x: x
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
        n_input_channels = 1
        num_classes_rec = 1
        base_num_features_reg = 16  # Half the number of parameters, double the fun
        self.network_reg = Generic_UNet(
            n_input_channels,
            base_num_features_reg,
            1,  # num classes=1 hardcoded, we just prod 1 image
            len(self.net_num_pool_op_kernel_sizes_reg),
            self.conv_per_stage_reg,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes_reg,
            self.net_conv_kernel_sizes_reg,
            False,
            True,
            True,
        )
        if torch.cuda.is_available():
            self.network_reg.cuda()
        # softmax helper seems to be rather harmless: lambda x: x
        self.network_reg.inference_apply_nonlin = softmax_helper

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
            self.num_input_channels,
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
