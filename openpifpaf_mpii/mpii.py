import argparse

import torch

import openpifpaf

from openpifpaf.plugins.coco.dataset import CocoDataset
from .constants import (
    CATEGORIES,
    KEYPOINTS,
    SKELETON,
    SIGMAS,
    SCORE_WEIGHTS,
    UPRIGHT_POSE,
    DENSER_CONNECTIONS,
    HFLIP,
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class MPII(openpifpaf.datasets.DataModule, openpifpaf.Configurable):
    # cli configurable
    train_annotations = 'MPII/annotations/MPII_coco_style_anns_train.json'
    val_annotations = 'MPII/annotations/MPII_coco_style_anns_train.json'  # TODO Find a way to not run val during training. Currently using the train data as validation, because MPII has no validation set
    eval_annotations = val_annotations
    train_image_dir = 'MPII/images'
    val_image_dir = 'MPII/images'
    eval_image_dir = val_image_dir

    square_edge = 385
    with_dense = False
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 0.1

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    skeleton = SKELETON

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cif = openpifpaf.headmeta.Cif('cif', 'mpii',
                                      keypoints=KEYPOINTS,
                                      sigmas=SIGMAS,
                                      pose=UPRIGHT_POSE,
                                      draw_skeleton=self.skeleton,
                                      score_weights=SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'mpii',
                                      keypoints=KEYPOINTS,
                                      sigmas=SIGMAS,
                                      pose=UPRIGHT_POSE,
                                      skeleton=self.skeleton)
        dcaf = openpifpaf.headmeta.Caf('caf25', 'mpii',
                                       keypoints=KEYPOINTS,
                                       sigmas=SIGMAS,
                                       pose=UPRIGHT_POSE,
                                       skeleton=DENSER_CONNECTIONS,
                                       sparse_skeleton=self.skeleton,
                                       only_in_field_of_view=True)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        dcaf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf, dcaf] if self.with_dense else [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module MPII Pose')

        group.add_argument('--mpii-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--mpii-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--mpii-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--mpii-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--mpii-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.with_dense
        group.add_argument('--mpii-with-dense',
                           default=False, action='store_true',
                           help='train with dense connections')
        assert not cls.extended_scale
        group.add_argument('--mpii-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--mpii-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--mpii-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--mpii-no-augmentation',
                           dest='mpii_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--mpii-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--mpii-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--mpii-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--mpii-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--mpii-eval-test2017', default=False, action='store_true')
        eval_set_group.add_argument('--mpii-eval-testdev2017', default=False, action='store_true')

        assert cls.eval_annotation_filter
        group.add_argument('--mpii-no-eval-annotation-filter',
                           dest='mpii_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--mpii-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--mpii-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--mpii-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # mpii specific
        cls.train_annotations = args.mpii_train_annotations
        cls.val_annotations = args.mpii_val_annotations
        cls.eval_annotations = args.mpii_val_annotations
        cls.train_image_dir = args.mpii_train_image_dir
        cls.val_image_dir = args.mpii_val_image_dir
        cls.eval_image_dir = args.mpii_val_image_dir

        cls.square_edge = args.mpii_square_edge
        cls.with_dense = args.mpii_with_dense
        cls.extended_scale = args.mpii_extended_scale
        cls.orientation_invariant = args.mpii_orientation_invariant
        cls.blur = args.mpii_blur
        cls.augmentation = args.mpii_augmentation
        cls.rescale_images = args.mpii_rescale_images
        cls.upsample_stride = args.mpii_upsample
        cls.min_kp_anns = args.mpii_min_kp_anns
        cls.bmin = args.mpii_bmin

        # evaluation
        cls.eval_annotation_filter = args.mpii_eval_annotation_filter
        if args.mpii_eval_test2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._test2017_annotations
            cls.annotation_filter = False
        if args.mpii_eval_testdev2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._testdev2017_annotations
            cls.annotation_filter = False
        cls.eval_long_edge = args.mpii_eval_long_edge
        cls.eval_orientation_invariant = args.mpii_eval_orientation_invariant
        cls.eval_extended_scale = args.mpii_eval_extended_scale

        if (args.mpii_eval_test2017 or args.mpii_eval_testdev2017) \
                and not args.write_predictions and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
        )]
