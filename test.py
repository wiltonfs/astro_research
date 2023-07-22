import time
from star_logger import StarLogger

output_dir = 'outs1'
logger = StarLogger(output_dir)

time.sleep(3)
logger.log("Done")
