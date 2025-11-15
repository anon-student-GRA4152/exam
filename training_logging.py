'''
Make custom made dceorator to log the training loss from VAE in a separte logging file

Maybe add time stamps?

Not working great cause there's alreayd the tf decorator which messing up the training...
'''

class LogTraining:
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

        # get the name of the func we decorating and add the loss
        log_string = self.func.__name__ + " was called - loss: " + str(loss)
        
        # Open the logfile and append
        with open (LogTraining._logfile, 'a') as opened_file:
            # Now we log to the specified logfile
            opened_file.write(log_string + '\n')
        
        # return base func
        return self.func(*args)

