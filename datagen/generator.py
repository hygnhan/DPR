
from utils import *


if __name__=='__main__':
    args = parse()

    if args.data == 'colored_mnist':
        from data.colored_mnist import colored_mnist_gen
        colored_mnist_gen(args)
    else:
        raise SystemExit('Unknown dataset...')

    
