import time
from comet_ml import Experiment, Optimizer
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from models import MyCNN, resnet18, Net1T, Net1T_HPO
from utils import device, show_dl
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging


if __name__ == '__main__':
    project = Project()
    # our hyperparameters
    params = {
        'epochs': 10,
        'lr': 0.001,
        'batch_size': 32,
        'model': 'Net1T',
        'numHiddenLayers': 3,
        'hiddenSize': 512,
        'dropout': .0
    }
    print()
    logging.info(f'Using device={device} 🚀 🔥  ⚡')
    
    # get the datasets
    train_dl, val_dl, test_dl = get_dataloaders(
        project.data_dir / "train",
        project.data_dir / "val",
        batch_size=params['batch_size'],
        pin_memory=True,
        num_workers=4,
    )
    
#     # define the algorithm and hyperparameters to use:
#     config = {
#         'algorithm': 'bayes',
#         'name': 'optimizer-search-0',
#         "spec": {"maxCombo": 20, "objective": "minimize", "metric": "loss"},
#         'parameters': {
#             'hiddenSize': {'type': 'integer', 'min': 128, 'max': 512},
#             'dropout': {'type': 'float', 'min': .0, 'max': 1.}
#         },
#         'trials': 1,
#     }
    
    # define the algorithm and hyperparameters to use:
    config = {
        'algorithm': 'grid',
        'name': 'optimizer-search-0',
        "spec": {"maxCombo": 9, "metric": "loss"},
        'parameters': {
            'hiddenSize': {'type': 'discrete', 'values': [64, 128, 256]},
            'dropout': {'type': 'discrete', 'values': [0.00, 0.125, 0.25]}
        },
        'trials': 1,
    }
    
    # instantiate optimizer object
    print('')
    opt = Optimizer(config, api_key="7bXV3NLiVQKVKcTtfx8jI03Vn")
    
    # hyper parameter search
    
    for experiment in opt.get_experiments(project_name="cometHPO"):
        
        # build the model
        inpSize=666; outSize=5; numHiddenLayers=params['numHiddenLayers'];
        net = Net1T_HPO(inpSize, outSize, numHiddenLayers, experiment).to(device)
        
        # define custom optimizer and instantiace the trainer `Model`
        optimizer = optim.Adam(net.parameters(), lr=params['lr'])
        model = Model(net, optimizer, "MSELoss", batch_metrics=["MSELoss"]).to(device)
        
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=True),
            ModelCheckpoint(str(project.checkpoint_dir / f"{time.time()}-model.pt"),
                            save_best_only="True",
                            verbose=True),
            EarlyStopping(monitor="val_loss", patience=10, mode='min'),
            CometCallback(experiment)
        ]
        model.fit_generator(
            train_dl,
            val_dl,
            epochs=params['epochs'],
            callbacks=callbacks
        )
        
        experiment.end()
