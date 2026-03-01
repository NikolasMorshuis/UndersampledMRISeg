from nnunet.network_architecture.generic_modular_UNet import (
    PlainConvUNetEncoder,
    PlainConvUNetDecoder,
    get_default_network_config,
)
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import StackedConvLayers
from nnunet.network_architecture.generic_UNet import Upsample
from torch import nn
import numpy as np


class One_Encoder_Two_Decoders(SegmentationNetwork):
    use_this_for_batch_size_computation_2D = 1167982592.0
    use_this_for_batch_size_computation_3D = 1152286720.0

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_blocks_per_stage_encoder,
        feat_map_mul_on_downscale,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        props,
        num_classes,
        num_blocks_per_stage_decoder,
        deep_supervision=True,
        upscale_logits=False,
        max_features=512,
        initializer=None,
        do_ds=True,
    ):
        super().__init__()
        self.conv_op = props["conv_op"]
        self.num_classes = num_classes

        self.encoder = PlainConvUNetEncoder(
            input_channels,
            base_num_features,
            num_blocks_per_stage_encoder,
            feat_map_mul_on_downscale,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            props,
            default_return_skips=True,
            max_num_features=max_features,
        )
        self.decoder_rec = PlainConvUNetDecoderRec(
            self.encoder, base_num_features, None, props, True, upscale_logits
        )

        self.decoder_seg = PlainConvUNetDecoder(
            self.encoder, num_classes, None, props, deep_supervision, upscale_logits
        )
        if initializer is not None:
            self.apply(initializer)

        self.do_ds = do_ds

    def forward(self, x):
        skips = self.encoder(x)
        rec = self.decoder_rec(skips)

        # concat skips[0] and rec
        new_skips = list((skips[-1],))
        # append list new_skips to list rec
        new_skips = rec + new_skips

        seg = self.decoder_seg(new_skips)
        if self.do_ds:
            return rec[0][:, 0:1], seg
        else:
            return rec[0][:, 0:1], seg[0]


class PlainConvUNetDecoderRec(nn.Module):
    def __init__(
        self,
        previous,
        num_classes,
        num_blocks_per_stage=None,
        network_props=None,
        deep_supervision=False,
        upscale_logits=False,
    ):
        super(PlainConvUNetDecoderRec, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props["conv_op"] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props["conv_op"] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s"
                % str(self.props["conv_op"])
            )

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = (
            len(previous_stages) - 1
        )  # we have one less as the first stage here is what comes after the
        # bottleneck

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        prev_stage_channels = previous_stage_output_features[::-1][1:]

        # only used for upsample_logits
        cum_upsample = np.cumprod(
            np.vstack(self.stage_pool_kernel_size), axis=0
        ).astype(int)

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            self.tus.append(
                transpconv(
                    features_below,
                    features_skip,
                    previous_stage_pool_kernel_size[s + 1],
                    previous_stage_pool_kernel_size[s + 1],
                    bias=False,
                )
            )
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(
                StackedConvLayers(
                    2 * features_skip,
                    features_skip,
                    previous_stage_conv_op_kernel_size[s],
                    self.props,
                    num_blocks_per_stage[i],
                )
            )

            if deep_supervision and s != 0:
                seg_layer = self.props["conv_op"](
                    features_skip, prev_stage_channels[i], 1, 1, 0, 1, 1, False
                )
                if upscale_logits:
                    upsample = Upsample(
                        scale_factor=cum_upsample[s], mode=upsample_mode
                    )
                    self.deep_supervision_outputs.append(
                        nn.Sequential(seg_layer, upsample)
                    )
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.props["conv_op"](
            features_skip, num_classes, 1, 1, 0, 1, 1, False
        )

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips, gt=None, loss=None):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]  # this is the bottleneck

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                tmp = self.deep_supervision_outputs[i](x)
                if gt is not None:
                    tmp = loss(tmp, gt)
                seg_outputs.append(tmp)

        segmentation = self.segmentation_output(x)

        if self.deep_supervision:
            tmp = segmentation
            if gt is not None:
                tmp = loss(tmp, gt)
            seg_outputs.append(tmp)
            return seg_outputs[
                ::-1
            ]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation


if __name__ == "__main__":
    conv_op_kernel_sizes = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3))
    pool_op_kernel_sizes = ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2))
    patch_size = (256, 256)
    batch_size = 56
    unet = One_Encoder_Two_Decoders(
        4,
        32,
        (2, 2, 2, 2, 2, 2, 2),
        2,
        pool_op_kernel_sizes,
        conv_op_kernel_sizes,
        get_default_network_config(2, dropout_p=None),
        4,
        (2, 2, 2, 2, 2, 2),
        False,
        False,
        max_features=512,
    ).cuda()

    test_file = torch.rand((batch_size, 4, 256, 256)).cuda()
    rec, seg = unet(test_file)
    print(rec.shape, seg.shape)
