import argparse

parser = argparse.ArgumentParser(description='Flow_based Image Interpolation')

# data specifications 
parser.add_argument('--split', type=str, default='train',
                    help='image dataset directory')
# parser.add_argument('--valid_file_root', type=str, default='./valid_CREMI/',
#                     help='image dataset directory')
parser.add_argument('--file_root', type=str, default='../total_epfl',
                    help='image dataset directory')
parser.add_argument('--file_name', type=str, default='epfl_10',
                    help='image dataset directory')
parser.add_argument('--flist_root', type=str, default='./Flist_epfl/',
                    help='flist store image name')
parser.add_argument('--image_size', type=int, default=256,
                    help='image size used during training')
parser.add_argument('--num_traindata', type=int, default=2000,
                    help='image size used during training')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers used in data loader')

# model specifications 
parser.add_argument('--flow_model_name', type=str, default='FlowInterp',
                    help='model name')
parser.add_argument('--model_name', type=str, default='FlowFuse',
                    help='model name')
parser.add_argument('--trainer_name', type=str, default='fuse_trainer',
                    help='model name')

# optimization specifications 
# parser.add_argument('--lr', type=float, default=1e-5,
#                     help='learning rate for flow estimator')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for flow estimator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 in optimier')
parser.add_argument('--adv_weight', type=float, default=0.1,
                    help='loss weight for adversarial loss')
# parser.add_argument('--adv_weight', type=float, default=0.01,
#                     help='loss weight for adversarial loss for mc')
                    
# training specifications 
parser.add_argument('--Interp_pre_train', type=str, default='../FlowInterp_result/epfl_flow_store_model/FlowInterp_epfl_256/flow_00138.pth',
                    help='path to pretrained models')
# parser.add_argument('--pre_train', type=str, default='../FlowFuse_result/sec_epfl_fuse_store_model/FlowFuse_epfl_256/gen_00114.pth',
#                     help='path to pretrained models')
parser.add_argument('--pre_train', type=str, default='/home/chengyiran/FlowInpaint_update/FlowFuse_result/epfl_fuse_store_model/FlowFuse_epfl_10_256/gen_00110.pth',
                    help='path to pretrained models')
                    # /home/chengyiran/FlowInpaint_update/FlowFuse_result/epfl_fuse_store_model/FlowFuse_epfl_10_256/gen_00110.pth
parser.add_argument('--load_flow', type=str, default=True,
                    help='whether load pretrained flow models')
parser.add_argument('--epoch', type=int, default=5000,
                    help='the number of epoch for training')
parser.add_argument('--iterations', type=int, default=1e6,
                    help='the number of iterations for training')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size in each mini-batch')
parser.add_argument('--port', type=int, default=23224,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')
parser.add_argument('--save_dir', type=str, default='./FlowFuse_result/epfl_fuse_store_model',
                    help='directory for saving models and logs')
parser.add_argument('--save_freq', type=int, default=5e2,
                    help='frequency for saving models')
parser.add_argument('--niter', type=int, default=5e4,
                    help='frequency for saving models')
parser.add_argument('--niter_steady', type=int, default=31e4,
                    help='frequency for saving models')
parser.add_argument('--img_save_dir', type=str, default='./FlowFuse_result/epfl_fuse_result',
                    help='directory for saving models and logs')
parser.add_argument('--verbosity', type=int, default= 2,
                    help='')
parser.add_argument('--img_save_freq', type=int, default= 20,
                    help='')
parser.add_argument('--valid_output_dir', type=str, default='./epfl_fuse_valid_result_4',
                    help='directory for saving models and logs')
args = parser.parse_args()
args.iterations = int(args.iterations)