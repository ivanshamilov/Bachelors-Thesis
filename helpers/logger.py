from datetime import datetime


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class Logger(object):

    @staticmethod
    def write_logs(to_write):
        with open("nn_diploma.logs", "a") as f:
            f.write(f"{to_write}\n")

    @staticmethod
    def info(toprint, log=True, console=True):
        time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        if console:
            print(f"{bcolors.OKCYAN}[{time}] [INFO] {toprint}{bcolors.ENDC}")
        if log:
            Logger.write_logs(f"{bcolors.OKCYAN}[{time}] [INFO] {toprint}{bcolors.ENDC}")

    @staticmethod
    def ok(toprint, log=True, console=True):
        time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        if console:
            print(f"{bcolors.OKGREEN}[{time}] [OK] {toprint}{bcolors.ENDC}")
        if log:
            Logger.write_logs(f"{bcolors.OKGREEN}[{time}] [OK] {toprint}{bcolors.ENDC}")

    @staticmethod
    def warning(toprint, log=True, console=True):
        time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        if console:
            print(f"{bcolors.WARNING}[{time}] [WARN] {toprint}{bcolors.ENDC}")
        if log:
            Logger.write_logs(f"{bcolors.WARNING}[{time}] [WARN] {toprint}{bcolors.ENDC}")

    @staticmethod
    def fail(toprint, log=True, console=True):
        time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") 
        if console:
            print(f"{bcolors.FAIL}[{time}] [FAIL] {toprint}{bcolors.ENDC}")
        if log:
            Logger.write_logs(f"{bcolors.FAIL}[{time}] [FAIL] {toprint}{bcolors.ENDC}")
