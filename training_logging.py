from datetime import datetime


class LogTraining:

    ''' 
    My own dectorator that logs the training steps of a VAE model. 
    It records timestamp when the train method of VAE was called
    and the resulting training loss. Unfortunatelly the training loss 
    gets returned in some odd tensorflow format that I was not able to
    extract the actual loss number from, so it only logs the tensorflow
    loss object, not actual training loss. But it still showcases both
    how to build a decorator and its purpose.
    '''

    # default file where the training loss will be logged
    _logfile = 'training_loss.log'

    def __init__ (self, func):
        self.func = func

    # mutator method, in case we want the training loss to be logged in a custom named file
    def setLogFile(file_name):
        LogTraining._logfile = file_name
    
    def __call__(self, *args):
        # the train func we decorating is returning loss so capture it here
        loss = self.func(*args)

        # get a timestamp when the method we are decorating was called
        timestamp = datetime.now()

        # get the method name
        func_name = self.func.__name__

        # add the timestamp, method name and loss together to form the log
        log_string = str(timestamp) + ' VAE ' + func_name + " was called - loss: " + str(loss)

        # Open the logfile and append
        with open (LogTraining._logfile, 'a') as opened_file:
            # Now we log to the specified logfile
            opened_file.write(log_string + '\n')
        
        # return base func
        return self.func(*args)

