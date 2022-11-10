import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_cifar.')


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayers_gradweight', '--2gw', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', '--2gi', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')
parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['exact', 'qat', 'all8bit', 'star_weight', 'only_weight', 'weight4', 'all4bit', 'forward8',
                             'forward4', 'plt'])
parser.add_argument('--choice', nargs='+', type=str, help='Choose a linear layer to quantize')

parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--task', type=str, default='mrpc', help='apply LSQ')
parser.add_argument('--seed', type=int, default=27, help='apply LSQ')

args = parser.parse_args()


arg = " -c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"
if args.training_bit == 'all8bit':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, awbits = 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, awbits = 0, 0, 8
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'star_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training_bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 8
    arg = "-c quantize --qa=False --qw=False --qg=True"
elif args.training_bit == 'weight4':
    bbits, bwbits, awbits = 8, 4, 8
elif args.training_bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, awbits = 8, 8, 4
else:
    bbits, bwbits, awbits = 0, 0, 0
    if args.training_bit == 'plt':
        arg = "-c quantize --qa={} --qw={} --qg={}".format(args.plt_bit[0], args.plt_bit[1], args.plt_bit[2])
        bbits, bwbits, awbits = args.plt_bit[5], args.plt_bit[4], args.plt_bit[3]

if args.twolayers_gradweight:
    assert bwbits == 4
if args.twolayers_gradinputt:
    assert bbits == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'full'

argchoice = ''
for cho in args.choice:
    argchoice += cho + ' '
    
os.system("accelerate launch test_glue.py --model_name_or_path bert-base-cased --task_name {} --max_length 128 "
          "--per_device_train_batch_size 32 --learning_rate 2e-5 --seed {} --num_train_epochs {} "
          "--output_dir ./test_glue_result_quantize/mrpc/quantize/seed={} --arch bertForSequence {} --choice {} "
          "--bbits {} --bwbits {} --abits {} --wbits {} "
          "--2gw {} --2gi {}"
          .format(args.task, args.seed, args.epochs, args.seed, arg, argchoice,
                  bbits, bwbits, awbits, awbits,
                  args.twolayers_gradweight, args.twolayers_gradinputt))
