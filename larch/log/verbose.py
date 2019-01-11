import logging

# create logger
logger = logging.getLogger('L4')
logger.setLevel(logging.DEBUG)

# create file handler which logs info messages
fh = logging.FileHandler('larch4.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create console stream formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s] %(name)s.%(levelname)s: %(message)s')
ch.setFormatter(formatter)

# create file stream formatter and add it to the handlers
# don't log time to file, facilitates diffs
formatter2 = logging.Formatter('%(name)s.%(levelname)s: %(message)s')
fh.setFormatter(formatter2)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

log = logger
