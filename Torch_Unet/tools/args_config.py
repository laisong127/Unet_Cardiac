import argparse

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--local_rank", type=int, default=0, required=False, help="device_ids of DistributedDataParallel")
parser.add_argument("--batch_size", type=int, default=4, required=False, help="batch size for training")
parser.add_argument("--epochs", type=int, default=300, required=False, help="Number of epochs to train for")
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader') #4
parser.add_argument("--ckpt_save_interval", type=int, default=50, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default='../../result/result_backbone0_1/10', help='specify an output directory if needed')
parser.add_argument('--mgpus', type=str, default=None, help='whether to use multiple gpu')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument('--train_with_eval', action='store_true', default=True, help='whether to train with evaluation')

args = parser.parse_args()
