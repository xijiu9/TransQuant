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
parser.add_argument('--forward-method', default='PTQ', type=str, metavar='strategy',
                    choices=['PTQ', 'LSQ', 'LSQplus', 'SAWB'])
parser.add_argument('--ACT2FN', type=str, default='gelu', help='')
parser.add_argument('--luq', type=str2bool, default=False, help='use luq for backward')
parser.add_argument('--training-bit', type=str, default='', help='weight number of bits', required=True,
                    choices=['exact', 'qat', 'all8bit', 'star_weight', 'only_weight', 'weight4', 'all4bit', 'forward8',
                             'forward4', 'plt'])
parser.add_argument('--plt-bit', type=str, default='', help='')
parser.add_argument('--choice', nargs='+', type=str, required=True, help='Choose a linear layer to quantize')

parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--task', type=str, default='mrpc', help='apply LSQ')
parser.add_argument('--seed', type=int, default=27, help='apply LSQ')
parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader.",)

parser.add_argument('--cutood', type=int, default=0, help='Choose a linear layer to quantize')
parser.add_argument('--clip-value', type=float, default=0, help='Choose a linear layer to quantize')
parser.add_argument('--plt-debug', type=str2bool, default=False, help='Debug to draw the variance and leverage score')

args = parser.parse_args()


arg = " -c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"
if args.training_bit == 'all8bit':
    bbits, bwbits, abits, wbits = 8, 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, abits, wbits = 0, 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, abits, wbits = 0, 0, 8, 9
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'star_weight':
    bbits, bwbits, abits, wbits = 8, 4, 4, 4
elif args.training_bit == 'only_weight':
    bbits, bwbits, abits, wbits = 8, 4, 8, 8
    arg = "-c quantize --qa=False --qw=False --qg=True"
elif args.training_bit == 'weight4':
    bbits, bwbits, abits, wbits = 8, 4, 8, 8
elif args.training_bit == 'all4bit':
    bbits, bwbits, abits, wbits = 4, 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, abits, wbits = 4, 4, 8, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, abits, wbits = 8, 8, 4, 4
else:
    bbits, bwbits, abits, wbits = 0, 0, 0, 0
    if args.training_bit == 'plt':
        arg = "-c quantize --qa={} --qw={} --qg={}".format(args.plt_bit[0], args.plt_bit[1], args.plt_bit[2])
        bbits, bwbits, abits, wbits = args.plt_bit[6], args.plt_bit[5], args.plt_bit[3], args.plt_bit[4]

if args.twolayers_gradweight:
    assert bwbits == 4
if args.twolayers_gradinputt:
    assert bbits == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = args.training_bit
    if args.luq:
        method = 'luq'
else:
    method = 'unknown'
    print("!"*1000)

argchoice = ''
arg_choice_without_space = ''
for cho in args.choice:
    argchoice += cho + ' '
    arg_choice_without_space += cho + '_'
argchoice = argchoice[:-1]

def op_None(x):
    return "None" if x is None else x
os.system("accelerate launch test_glue.py --model_name_or_path bert-base-cased --task_name {} --max_length 128 "
          "--per_device_train_batch_size {} --learning_rate 2e-5 --seed {} --num_train_epochs {} "
          "--output_dir ./test_glue_result_quantize/{}/{}/choice={}/seed={} --arch BertForSequenceClassification {} --choice {} "
          "--bbits {} --bwbits {} --abits {} --wbits {} "
          "--2gw {} --2gi {} --luq {} --forward-method {}"
          " --cutood {} --clip-value {} --ACT2FN {} "
          "--plt-debug {}"
          .format(args.task, args.per_device_train_batch_size, args.seed, args.epochs,
                  args.task, method, arg_choice_without_space, args.seed, arg, argchoice,
                  bbits, bwbits, abits, wbits,
                  args.twolayers_gradweight, args.twolayers_gradinputt, args.luq, args.forward_method,
                  args.cutood, args.clip_value, args.ACT2FN,
                  args.plt_debug))
