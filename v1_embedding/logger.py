import datetime
import sys
import os

class MyLogger:
    def __init__(self, stdout, filename):
        self.stdout = stdout
        self.logfile = open(filename, 'w')

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)

    def close(self):
        pass
        # self.stdout.close()
        # self.logfile.close()

    def flush(self):
        self.logfile.flush()


def init_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_file_name = 'logs/{}.log'.format(now)
    writer = MyLogger(sys.stdout, log_file_name)
    sys.stdout = writer
    sys.stderr = writer
    print('Saving logs both to stdout and {}'.format(log_file_name))
