#!/usr/bin/env python

from nn_training.handler import main_nn_handler
from log_collecting.log_collector import *
from helpers.logger import Logger as logger
from log_collecting.transformer import transformer_handler

import sys
import os


def collect_logs():
    args = enable_parser()
    sanity_check(args)
    create_pcap(args)
    preprocess_pcap(args)
    return args.dir, args.filename


def main():
    args = enable_parser()
    sanity_check(args)

    if args.mode == "train":
        logger.info("Running script in training mode...")
        filenames = [filename for filename in os.listdir(args.dir)]
        main_nn_handler(folder_path=args.dir, filenames=filenames)
    elif args.mode == "real-time":
        logger.info("Running script in real-time mode...")
        for _ in iter(int, 1):
            create_pcap(args)
            preprocess_pcap(args)
            outdir, file = args.dir, args.file
            file = f"{outdir}/{file}.csv"
            transformer_handler(file)


if __name__ == "__main__":
    logger.info("==========================RUN INITIATED==========================", console=False)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("===========================RUN FINISHED BY INTERRUPT===========================", console=False)
        sys.exit(0)
    logger.info("===========================RUN FINISHED===========================", console=False)
