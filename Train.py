from torch.utils.data import DataLoader
from KNN import DeepKnn
from Loader import load_skin_datasets
from Models import VGG19

PATHS = {'images_path': '../ISIC-2017_Training_Data',
         'labels': 'ISIC-2017_Training_Part3_GroundTruth.csv'}
SETTING = {'optimizer': 'adam',
           'lr': 0.001,
           'epochs': 100,
           'batch_size': 128}

if __name__ == '__main__':
    train, test = load_skin_datasets(PATHS['images_path'], PATHS['labels'])
    train_loader = DataLoader(dataset=train, batch_size=SETTING['batch_size'])
    test_loader = DataLoader(dataset=test, batch_size=SETTING['batch_size'])

    net = VGG19().cuda()
    model = DeepKnn()
    model.train(train_loader, net, SETTING['lr'], SETTING['batch_size'], test_loader)