# # Log model progress
# Felix Wilton
# 7/22/2023
import time
class StarLogger:
    def __init__(self, logging_dir):
        self.file = open(logging_dir + '\log.txt', 'a')
        self.start = time.time()
        self.log('Created logger')


    def log(self, msg):
        msg = '[' + str(time.time() - self.start) + ']\t' + msg
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()  # Flush the buffer to ensure data is written immediately

    def close(self):
        self.file.close()
