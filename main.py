import argparse
import city
import lidc

parser = argparse.ArgumentParser(description='Probabilistic U-Net')
parser.add_argument('dataset', type=str,
                    help='Type of dataset, "city" for Cityscape, "lidc" for LIDC dataset')
parser.add_argument('-b', '--batch-size', type=int, help='Batch size for train and val', required=True)
parser.add_argument('--val-after', type=int, help='Run validation after this No of iterations', required=True)
parser.add_argument('-e', '--epoch', type=int, help='Number of epochs', required=True)
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--lr', type=float, help='Learning rate', required=True)


args = parser.parse_args()

print(args)

if args['dataset'] == 'city':
    city.train(batch_size=args['batch_size'], val_after=args['val_after'], lr=args['lr'], gpu=args['gpu'])
    city.test(gpu=args['gpu'])

elif args['dataset'] == 'lidc':
    lidc.train(batch_size=args['batch_size'], val_after=args['val_after'], lr=args['lr'], gpu=args['gpu'])
    lidc.test(gpu=args['gpu'])

else:
    print(f"No dataset called {args['dataset']}")
