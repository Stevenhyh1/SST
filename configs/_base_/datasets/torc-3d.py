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
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "kitti_infos_train.pkl",
            split="training",
            pts_prefix="velodyne_reduced",
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
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        pcd_limit_range=point_cloud_range
    ),
)
evaluation = dict(interval=1, pipeline=test_pipeline)
