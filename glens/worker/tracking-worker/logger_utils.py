import  coloredlogs, logging


COLORED_LOGS_LEVEL_STYLES={'performance':{'color':'blue'},'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}


# Install the coloredlogs module on the root logger
coloredlogs.install(level_styles=COLORED_LOGS_LEVEL_STYLES)



### add level to log performance data also
PERFORMANCE_LEVELV_NUM = 1
logging.addLevelName(PERFORMANCE_LEVELV_NUM, "PERFORMANCE")
setattr(logging, "PERFORMANCE", PERFORMANCE_LEVELV_NUM )

def performance(self, message, *args, **kws):
    if self.isEnabledFor(PERFORMANCE_LEVELV_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(PERFORMANCE_LEVELV_NUM, message, args, **kws)
logging.Logger.performance = performance

def init_logger(name, isDevelopment=True):
     
    logger=logging.getLogger(name)
    fh=logging.FileHandler("logs_"+("development" if  isDevelopment else "production")+".log")
    #fh.setLevel(level=logging.DEBUG if isDevelopment else logging.WARNING)
    fh.setLevel(level=logging.DEBUG if isDevelopment else logging.PERFORMANCE)
    formatter = coloredlogs.ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',level_styles=COLORED_LOGS_LEVEL_STYLES)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG) 
    return logger
    
