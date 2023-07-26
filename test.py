import time
from star_logger import StarLogger

logger = StarLogger("test/")

logger.log("Sleep for 3 seconds")
time.sleep(3)
logger.log("Done")
