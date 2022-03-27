import argparse
parser = argparse.ArgumentParser()

parser.add_argument("image_path", help="image_path: the string to the store the path of image")
parser.add_argument("model_file", help="model_name: the string to the store the file name of model")

parser.add_argument("--top_k", type=int, help="optional argument, the top K highest probablity classes will be shown")
parser.add_argument("--category_names",  help="optional argument, the flower names will be shown")

args = parser.parse_args()

image_path=args.image_path
model_file=args.model_file

if args.top_k:
    top_k=args.top_k
else:
    top_k=False

if args.category_names:
    category_names=args.category_names
else:
    category_names=False
