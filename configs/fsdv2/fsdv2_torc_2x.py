# dataset settings
data_root = "data/kitti/"
dataset_type = "TorcDataset"
file_client_args = dict(backend="disk")


class_names = [
    "Dynamic--Vehicle--Passenger--Car",
    "Dynamic--Vehicle--SemiTruck--Trailer",
    "Dynamic--Vehicle--SemiTruck--Cab",
    "Dynamic--Vehicle",
    "Static--RoadObstruction--Barrel",
    "Static--RoadObstruction--TemporaryBarrier",
    "Static--RoadObstruction--Cone",
]
point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "kitti_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points={
            "Dynamic--Vehicle--Passenger--Car": 5,
            "Dynamic--Vehicle--SemiTruck--Trailer": 5,
            "Dynamic--Vehicle--SemiTruck--Cab": 5,
            "Dynamic--Vehicle": 5,
            "Static--RoadObstruction--Cone": 5,
            "Static--RoadObstruction--Barrel": 5,
            "Static--RoadObstruction--TemporaryBarrier": 5,
        },
    ),
    classes=class_names,
    sample_groups={
        "Dynamic--Vehicle--Passenger--Car": 20,
        "Dynamic--Vehicle--SemiTruck--Trailer": 20,
        "Dynamic--Vehicle--SemiTruck--Cab": 20,
        "Dynamic--Vehicle": 20,
        "Static--RoadObstruction--Cone": 15,
        "Static--RoadObstruction--Barrel": 15,
        "Static--RoadObstruction--TemporaryBarrier": 15,
    },
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args,
    ),
    dict(type="ObjectSample", db_sampler=db_sampler),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.785, 0.785],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizePoints"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="NormalizePoints"),
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="Collect3D",
                keys=["points"],
            ),
        ],
    ),
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "kitti_infos_train.pkl",
            split="training",
            pts_prefix="velodyne",
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d="LiDAR",
            pcd_limit_range=point_cloud_range,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "kitti_infos_val.pkl",
        split="testing",
        pts_prefix="velodyne",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        pcd_limit_range=point_cloud_range
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "kitti_infos_val.pkl",
        split="testing",
        pts_prefix="velodyne",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        pcd_limit_range=point_cloud_range
    ),
)

seg_voxel_size = (0.4, 0.4, 0.4)
virtual_voxel_size = (0.4, 0.4, 0.4)  # (1024, 1024, 16)

sparse_shape = [32, 2048, 2048]


group1 = ["Dynamic--Vehicle--Passenger--Car"]
group2 = ["Dynamic--Vehicle--SemiTruck--Trailer"]
group3 = ["Dynamic--Vehicle--SemiTruck--Cab"]
group4 = ["Dynamic--Vehicle"]
group5 = [
    "Static--RoadObstruction--Barrel",
    "Static--RoadObstruction--TemporaryBarrier",
    "Static--RoadObstruction--Cone",
]
num_classes = len(class_names)
group_names = [group1, group2, group3, group4, group5]

seg_score_thresh = [0.4, 0.25, 0.25, 0.25, 0.25]
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5)]

segmentor = dict(
    type="VoteSegmentor",
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1),
    ),
    voxel_encoder=dict(
        type="DynamicScatterVFE",
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
    ),
    middle_encoder=dict(
        type="PseudoMiddleEncoderForSpconvFSD",
    ),
    backbone=dict(
        type="SimpleSparseUNet",
        in_channels=64,
        sparse_shape=sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,  # dummy
        encoder_channels=(
            (128,),
            (128, 128),
            (128, 128),
            (128, 128, 128),
            (256, 256, 256),
            (256, 256, 256),
        ),
        encoder_paddings=(
            (1,),
            (1, 1),
            (1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        decoder_channels=(
            (256, 256, 256),
            (256, 256, 128),
            (128, 128, 128),
            (128, 128, 128),
            (128, 128, 128),
            (128, 128, 128),
        ),
        decoder_paddings=(
            (1, 1),
            (1, 0),
            (1, 0),
            (0, 0),
            (0, 1),
            (1, 1),
        ),  # decoder paddings seem useless in SubMConv
        return_multiscale_features=True,
    ),
    decode_neck=dict(
        type="Voxel2PointScatterNeck",
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    segmentation_head=dict(
        type="VoteSegHead",
        in_channel=131,
        hidden_dims=[128, 128],
        num_classes=len(class_names),
        dropout_ratio=0.0,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="naiveSyncBN1d"),
        act_cfg=dict(type="ReLU"),
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            class_weight=[1.0] * len(class_names) + [0.1],
            loss_weight=3.0,
        ),
        loss_vote=dict(type="L1Loss", loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh,  # no
        class_names=class_names,
        group_names=group_names,
        group_lens=group_lens,
    ),
)

model = dict(
    type="SingleStageFSDV2",
    segmentor=segmentor,
    virtual_point_projector=dict(
        in_channels=len(class_names) + 72 + 64,
        hidden_dims=[64, 64],
        norm_cfg=dict(type="naiveSyncBN1d"),
        ori_in_channels=67 + 64,
        ori_hidden_dims=[64, 64],
        # recover_in_channels=128 + 3, # with point2voxel offset
        # recover_hidden_dims=[128, 128],
    ),
    multiscale_cfg=dict(
        multiscale_levels=[0, 1, 2],
        projector_hiddens=[[256, 128], [128, 128], [128, 128]],
        fusion_mode="avg",
        target_sparse_shape=[16, 1024, 1024],
        norm_cfg=dict(type="naiveSyncBN1d"),
    ),
    voxel_encoder=dict(
        type="DynamicScatterVFE",
        in_channels=67,
        feat_channels=[64, 128],
        voxel_size=virtual_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        unique_once=True,
    ),
    backbone=dict(
        type="VirtualVoxelMixer",
        in_channels=128,
        sparse_shape=[16, 1024, 1024],
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64,), (64, 64), (64, 64)),
        encoder_paddings=((1,), (1, 1), (1, 1)),
        decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 1), (1, 1)),
    ),
    bbox_head=dict(
        type="FSDV2Head",
        num_classes=num_classes,
        bbox_coder=dict(type="BasePointBBoxCoder"),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=1.0, alpha=0.25, loss_weight=4.0
        ),
        loss_center=dict(type="SmoothL1Loss", loss_weight=0.25, beta=0.1),
        loss_size=dict(type="SmoothL1Loss", loss_weight=0.25, beta=0.1),
        loss_rot=dict(type="SmoothL1Loss", loss_weight=0.1, beta=0.1),
        in_channel=128,
        shared_mlp_dims=[256, 256],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type="LN"),
        tasks=[
            dict(class_names=group1),
            dict(class_names=group2),
            dict(class_names=group3),
            dict(class_names=group4),
            dict(class_names=group5),
        ],
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128),
            dim=(3, 2, 128),
            rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type="FSDSeparateHead",
            norm_cfg=dict(type="LN"),
            act="relu",
        ),
    ),
    train_cfg=dict(
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        batched_group_sample=True,
        offset_weight="max",
        class_names=class_names,
        group_names=[group1, group2, group3, group4, group5],
        centroid_assign=True,
        disable_pretrain=True,
        disable_pretrain_topks=[300] * num_classes,
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        batched_group_sample=True,
        offset_weight="max",
        class_names=class_names,
        group_names=[group1, group2, group3, group4, group5],
        use_rotate_nms=True,
        nms_pre=-1,
        nms_thr=0.15,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500,
    ),
)

lr=1e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=3, pipeline=test_pipeline)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
