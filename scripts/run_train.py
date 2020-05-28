import yaml
from addict import Dict
from scripts.train import argument_parser
from scripts.train import main as train_main
import torch
import shutil
import os
import random
import sys


# needed to solve tuple reading issue
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def main():
    # argument: name of yaml config file
    try:
        filename = sys.argv[1]
    except IndexError as error:
        print("provide yaml file as command-line argument!")
        exit()

    config = Dict(yaml.load(open(filename), Loader=PrettySafeLoader))
    os.makedirs(config.log_dir, exist_ok=True)
    # save a copy in the experiment dir
    shutil.copyfile(filename, os.path.join(config.log_dir, 'args.yaml'))

    torch.cuda.set_device(config.gpu)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    args = yaml_to_parser(config)
    train_main(args)


def yaml_to_parser(config):
    parser = argument_parser()
    args, unknown = parser.parse_known_args()

    args_dict = vars(args)
    for key, value in config.items():
        try:
            args_dict[key] = value
        except KeyError:
            print(key, ' was not found in arguments')
    return args


if __name__ == '__main__':
    main()
