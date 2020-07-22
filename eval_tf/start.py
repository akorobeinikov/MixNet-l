import tensorflow.compat.v1 as tf
import eval
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="name of model", default="mixnet-l")
parser.add_argument("--model", help="path to model", default="../mixnet-l/TF")
parser.add_argument("--image", help="path to image", default="C:/images/panda.jpg")
parser.add_argument("--labels", help="path to image", default="../labels.json")

args = parser.parse_args()

def main():
    print(tf.__version__)
    driver = eval.EvalCkptDriver(args.name)
    output = driver.eval_example_images(args.model, [args.image], args.labels)

if __name__ == '__main__':
    main()
