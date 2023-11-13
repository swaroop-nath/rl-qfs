import configparser

HYPERPARAM_PATH = 'hyperparams_rl_grad.cfg'

class ConfigManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        with open(HYPERPARAM_PATH, 'r') as file:
            self.config.read_file(file)
        self.config = self.config['ALL_PARAMS']
        self.kwargs = {}

        for key, value in self.config.items():
            self.kwargs[key] = eval(value)

    def get_kwargs(self):
        return self.kwargs