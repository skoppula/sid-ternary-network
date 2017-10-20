from cifar10_resnet_with_tracking import ResnetModel
from tensorpack.tfutils.export import ModelExport
import os
import argparse
from helpers import _mkdir

def export_model(in_dir, out_dir, use_latest, n):

    if use_latest:
        if use_latest == 'true' or use_latest is True:
            use_latest = True
        elif use_latest == 'false' or use_latest is False:
            use_latest = False
        else: assert False
    else:
        use_latest = True

    if use_latest:
        ckpt_file = os.path.join(in_dir, 'checkpoint')
    else:
        ckpt_file = os.path.join(in_dir, 'min-validation_error')
    print("Using", ckpt_file)

    # _mkdir(out_dir)

    e = ModelExport(ResnetModel(n=int(n)), ['input'], ['output'])
    e.export(ckpt_file, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', help='where to output exported model.')
    parser.add_argument('--in_dir', help='dir where to find exported model files')
    parser.add_argument('--use_latest', help='true|false')
    parser.add_argument('--n', help='true|false')
    args = parser.parse_args()

    export_model(args.in_dir, args.out_dir, args.use_latest, int(args.n))
