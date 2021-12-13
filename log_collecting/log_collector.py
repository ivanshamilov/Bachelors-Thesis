import subprocess
import os
import argparse
import psutil
import time
import signal

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from helpers.logger import Logger as logger


def countdown(interval):
    for _ in tqdm(range(interval), desc="Collecting data"):
        time.sleep(1)


def enable_parser():
    parser = argparse.ArgumentParser(description="Log gathering script")
    parser.add_argument('--mode', '-m', nargs="?", default="real-time", required=True, dest="mode", 
                help="Mode of the script: train – reading csv files and training the data on them, real-time – real time log collecting and flow predicting")
    parser.add_argument('--dir', '-d', metavar='output directory', dest='dir', nargs='?', default=os.getcwd(), 
                help='For train mode: directory where the csv data is stored, for real-time mode: the output directory of pcap and csv files', type=Path)
    parser.add_argument('--name', '-n', metavar='output filename', dest='filename', default="script_test", help='Output Filename', type=Path)
    parser.add_argument('--interface', '-i', metavar='interface', dest='interface', help='Interface')
    parser.add_argument('--capturing_interval', '-ci', metavar='Capturing Interval', nargs="?", default=60, type=int, dest='capt_interval', help='Capturing Interval')
    parser.add_argument('-f', dest="file", default=None, required=False)
    args = parser.parse_args()
    return args


def sanity_check(args):
    args.dir = os.path.expanduser(args.dir)
    if args.mode == "real-time":
        if not args.interface in psutil.net_if_addrs().keys():
            logger.fail("Provided interface does not exist.")
            raise Exception
        try:
            os.makedirs(args.dir)
        except OSError as _:
            logger.warning("Directory Exists. Creating files...")
    elif args.mode == "train":
        if not os.path.exists(args.dir):
            logger.fail(f"Provided path ({args.dir}) does not exist, please check and try again")


def create_pcap(args):
    args.file = str(args.filename) + "-" + datetime.now().strftime("%d.%m.%y-%H.%M.%S")
    tcpdump = f"sudo tcpdump -U -i {args.interface} -w {args.dir}/{args.file}.pcap"
    logger.info(f"Running tcpdump as {os.getlogin()} for {args.capt_interval} seconds")
    proc = subprocess.Popen(tcpdump, shell=True, stdout=subprocess.PIPE)
    time.sleep(0.1)
    countdown(args.capt_interval)
    proc.send_signal(signal.SIGTERM)
    logger.ok("tcpdump finished.")


def preprocess_pcap(args):
    logger.info(f"Running cicflowmeter as {os.getlogin()}")
    converter = f"cicflowmeter --file {args.dir}/{args.file}.pcap --csv {args.dir}/{args.file}.csv"
    proc = subprocess.call(converter, shell=True)
    logger.ok("cicflowmeter finished.")
