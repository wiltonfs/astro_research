# # Log model progress
# Felix Wilton
# 7/22/2023
import time
import os
class StarLogger:
    def __init__(self, logging_dir):
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        self.file = open(os.path.join(logging_dir, 'log.txt'), 'w')  # Use 'w' mode to create or overwrite the log file
        self.start = time.time()
        self.log('Created logger')


    def log(self, msg):
        current_time = time.time()
        time_diff = current_time - self.start
        formatted_time = f'[{time_diff:.2f}]\t'
        msg = formatted_time + msg
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()  # Flush the buffer to ensure data is written immediately

    def close(self):
        self.file.close()
