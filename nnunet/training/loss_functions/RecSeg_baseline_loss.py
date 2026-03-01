from torch import nn
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


class RecSeg_baseline_loss(nn.Module):
    def __init__(
        self,
        batch_dice=True,
        ds_loss_weights=None,
        seg_loss_weight=1.0,
        rec_loss_weight=1.0,
    ):
        super().__init__()
        self.seg_loss = MultipleOutputLoss2(
            DC_and_CE_loss(
                {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False}, {}
            ),
            ds_loss_weights,
        )
        self.rec_loss = nn.MSELoss()
        self.seg_loss_weight = seg_loss_weight
        self.rec_loss_weight = rec_loss_weight

    def forward(self, pred_rec, pred_seg, target_rec, target_seg):
        seg_loss = self.seg_loss(pred_seg, target_seg)
        rec_loss = self.rec_loss(pred_rec, target_rec)
        return rec_loss * self.rec_loss_weight + seg_loss * self.seg_loss_weight
