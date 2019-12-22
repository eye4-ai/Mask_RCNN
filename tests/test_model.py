import mrcnn.model as modellib
import numpy as np
import os.path
import pytest
import skimage.io
import urllib.request

from mrcnn.config import Config
from mrcnn import utils
from pathlib import Path


ROOT_DIR = Path()
CACHE_DIR = ROOT_DIR/"cache"
TEST_IMAGE_PATH = ROOT_DIR/'images/3627527276_6fe8cd9bfe_z.jpg'


class UnittestConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "unittest"
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    GPU_COUNT = 1


class UnittestDataset(utils.Dataset):
    """ A Dataset for testing """


@pytest.fixture
def model_data():
    """ Fixture for downloading mask_rcnn_coco training data
    """
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    test_model_path = str((CACHE_DIR / "mask_rcnn_coco.h5").resolve())
    if not os.path.isfile(test_model_path):
        urllib.request.urlretrieve(
            "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5",
            test_model_path)
    return test_model_path


def test_inference_detect(tmpdir, model_data):
    config = UnittestConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=tmpdir, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(model_data, by_name=True)
    image = skimage.io.imread(TEST_IMAGE_PATH)
    result = model.detect([image], verbose=1)[0]
    assert np.all([result['class_ids'], [24, 23, 23, 23]])
    assert np.all([np.greater(result['scores'], [0.99, 0.99, 0.99, 0.99]), [True, True, True, True]])


def x_test_training(tmpdir, model_data):
    LR = 1e-4
    EPOCHS = [2, 6, 8]
    config = UnittestConfig()
    model = modellib.MaskRCNN(mode='training', model_dir=tmpdir, config=config)
    model.load_weights(model_data, by_name=True, exclude=[
        'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # train_dataset = UnittestDataset(train_df)
    # train_dataset.prepare()
    #
    # valid_dataset = UnittestDataset(valid_df)
    # valid_dataset.prepare()
    #
    # model.train(train_dataset, valid_dataset,
    #             learning_rate=LR * 2,  # train heads with higher lr to speedup learning
    #             epochs=EPOCHS[0],
    #             layers='heads',
    #             augmentation=None)


