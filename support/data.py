from fastai.vision.all import *
import config
import random

def get_mask_path(photo):
    return photo.parent / f"{photo.stem}_{config.LABEL_ENDING}{photo.suffix}"

def get_validation(photo):
    return photo.stem.endswith(config.VALID_ENDING)

def get_images(files):
    image = []
    for file in files:
        if not file.stem.endswith(config.LABEL_ENDING):
            image.append(file)
    return image

def make_dls():
    files = get_image_files(config.DATA_PATH)
    photos = get_images(files)

    if config.DEBUG == True:
        train_photos = [
            photo for photo in photos
            if not get_validation(photo)
        ]

        valid_photos = [
            photo for photo in photos
            if get_validation(photo)
        ]

        random.seed(config.SEED)
        train_photos = random.sample(train_photos, min(config.TRAIN_IMAGES, len(train_photos)))
        valid_photos = random.sample(valid_photos, min(config.VALID_IMAGES, len(valid_photos)))

        photos = train_photos + valid_photos

    data = DataBlock(blocks=(ImageBlock, MaskBlock(codes=config.CODES)), get_y=get_mask_path, splitter=FuncSplitter(get_validation))

    return data.dataloaders(photos, bs=config.BATCH_SIZE, num_workers=config.WORKERS)
