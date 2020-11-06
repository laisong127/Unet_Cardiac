from libs.datasets import joint_augment, AcdcDataset
from libs.datasets.acdc_dataset_Isensee import AcdcDataset_Isensee
from libs.datasets.batchgenerators.transforms import SpatialTransform
from torch.utils.data import DataLoader
import numpy as np
from libs.datasets import augment as standard_augment
from libs.configs.config_acdc import cfg
from tools.args_config import args


def create_dataloader_Insensee(do_elastic_transform = True, alpha = (100., 350.),
                               sigma = (14., 17.), do_rotation = True,
                               a_x = (0., 2 * np.pi), a_y = (-0.000001, 0.00001),
                               a_z = (-0.000001, 0.00001),
                               do_scale = True, scale_range = (0.7, 1.3)):

    transform = SpatialTransform((352, 352), list(np.array((352, 352)) // 2), do_elastic_transform, alpha, sigma,
                                 do_rotation, a_x, a_y, a_z, do_scale, scale_range,
                                 'constant', 0, 3, 'constant',
                                 0, 0, random_crop=False)

    train_set_Isensee = AcdcDataset_Isensee(data_list=cfg.DATASET.TRAIN_LIST, Isensee_augment=transform)

    train_loader_Isensee = DataLoader(train_set_Isensee, batch_size=args.batch_size, pin_memory=True,
                                      num_workers=1, shuffle=False)

    if args.train_with_eval:
        eval_transform = joint_augment.Compose([
            joint_augment.To_PIL_Image(),
            joint_augment.FixResize(352),  # divided by 32
            joint_augment.To_Tensor()])
        evalImg_transform = standard_augment.Compose([
            standard_augment.normalize_meanstd()])

        if cfg.DATASET.NAME == 'acdc':
            test_set = AcdcDataset(data_list=cfg.DATASET.TEST_LIST,
                                   joint_augment=eval_transform,
                                   augment=evalImg_transform)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=True,
                                 num_workers=args.workers, shuffle=False,
                                 )
    else:
        test_loader = None

    return train_loader_Isensee, test_loader