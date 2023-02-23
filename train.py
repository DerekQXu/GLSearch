from options.train_options import TrainOptions
#from data.load_datasets import load_dataset_list

def train(opt):
    # Load datasets as list of CurriculumDataset objects.
    #dataset_list = load_dataset_list(opt.dataset_list, opt)
    print(opt.phase)
    pass

if __name__ == '__main__':
    options = TrainOptions()
    options.initialize()
    _opt = options.parse()
    train(_opt)