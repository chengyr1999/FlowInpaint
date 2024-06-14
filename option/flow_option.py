import argparse

parser = argparse.ArgumentParser(description='Flow_based Image Interpolation')

# data specifications 
parser.add_argument('--split', type=str, default='train',
                    help='image dataset directory')
parser.add_argument('--file_root', type=str, default='../total_cremi',
                    help='image dataset directory')
# parser.add_argument('--file_name', type=str, default='CREMI_B',
#                     help='image dataset directory')
# parser.add_argument('--file_name', type=str, default='CREMI_A',
#                     help='image dataset directory')
parser.add_argument('--file_name', type=str, default='CREMI_C',
                    help='image dataset directory')
parser.add_argument('--flist_root', type=str, default='./Flist_CREMI/',
                    help='flist store image name')
parser.add_argument('--image_size', type=int, default=256,
                    help='image size used during training')
parser.add_argument('--num_train_dataset', type=int, default=2000,
                    help='image num used during training')
parser.add_argument('--step', type=float, default=0.5,
                    help='image num used during training')
parser.add_argument('--num_traindata', type=int, default=2000,
                    help='image size used during training')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=8,
                    help='number of workers used in data loader')

# model specifications 
parser.add_argument('--model_name', type=str, default='FlowInterp',
                    help='model name')
parser.add_argument('--trainer_name', type=str, default='flow_trainer',
                    help='trainer name')

# optimization specifications 
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for flow estimator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 in optimier')

# training specifications 
parser.add_argument('--iterations', type=int, default=1e6,
                    help='the number of iterations for training')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size in each mini-batch')
parser.add_argument('--epoch', type=int, default=5000,
                    help='the number of epoch for training')
parser.add_argument('--port', type=int, default=12154,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')
parser.add_argument('--save_freq', type=int, default=1000,
                    help='frequency for saving models')
# parser.add_argument('--save_dir', type=str, default='../FlowInterp_result/A_flow_store_model_our',
#                     help='directory for saving models and logs')
# parser.add_argument('--img_save_dir', type=str, default='../FlowInterp_result/A_Interp_result_our',
#                     help='directory for saving models and logs')
# parser.add_argument('--save_dir', type=str, default='../FlowInterp_result/B_flow_store_model_our',
#                     help='directory for saving models and logs')
# parser.add_argument('--img_save_dir', type=str, default='../FlowInterp_result/B_Interp_result_our',
#                     help='directory for saving models and logs')
parser.add_argument('--save_dir', type=str, default='../FlowInterp_result/C_flow_store_model_our',
                    help='directory for saving models and logs')
parser.add_argument('--img_save_dir', type=str, default='../FlowInterp_result/C_Interp_result_our',
                    help='directory for saving models and logs')
parser.add_argument('--niter', type=int, default=10e4,
                    help='frequency for saving models')
parser.add_argument('--niter_steady', type=int, default=31e4,
                    help='frequency for saving models')
parser.add_argument('--verbosity', type=int, default= 2,
                    help='')
parser.add_argument('--img_save_freq', type=int, default= 25,
                    help='')
args = parser.parse_args()
args.iterations = int(args.iterations)