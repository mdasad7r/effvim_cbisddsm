import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, use_clahe: bool):
    pre = []
    if use_clahe:
        # Works best on grayscale. If input is 3-channel replicated grayscale, CLAHE still helps.
        pre.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

    train_tf = A.Compose(
        pre + [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_tf = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    tta_tf = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return train_tf, val_tf, tta_tf
