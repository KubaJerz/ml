class EEGDataScript(BaseDataScript):
    def get_data(self):
        # Use self.config to access parameters:
        train_split = self.config['train_split']
        seed = self.config['seed']
        shuffle = self.config['shuffle']
        # ... implementation ...