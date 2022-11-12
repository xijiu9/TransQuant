import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--method', type=str, default='', help='weight number of bits', required=True,
                    choices=['twolayer', 'luq', 'exact', 'all8bit', 'all4bit', 'forward8', 'forward4'])
parser.add_argument('--cover', type=str, default='', help='weight number of bits', required=True,
                    choices=['dataset', 'choices'])
args = parser.parse_args()

control_bit = ''
control_quant = ''
if args.method == 'twolayer':
    control_quant = "--choice quantize "
    control_bit = "--training-bit forward8 --2gw True --2gi True "
elif args.method == 'luq':
    control_quant = "--choice quantize "
    control_bit = '--training-bit forward8 --luq True'
elif args.method == 'exact':
    control_quant = "--choice classic "
    control_bit = '--training-bit all8bit'
elif args.method == 'all8bit' or args.method == 'all4bit' or args.method == 'forward8':
    control_quant = "--choice quantize "
    control_bit = '--choice quantize --training-bit {}'.format(args.method)





if args.cover == 'dataset':
    for task in ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli']:
        os.system("python test_mrpc.py {} {} --task {}".format(control_quant, control_bit, task))

elif args.cover == 'choices':
    for choices in ["embedding", "attention", "addNorm", "feedForward", "pooler", "classifier"]:
        control_quant = '--choice {}'.format(choices)
        os.system("python test_mrpc.py {}".format(control_quant, control_bit))
