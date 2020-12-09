import pandas as pd
import numpy as np
from .CustomDataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder


# def get_dataloaders(
#         train_dir,
#         val_dir,
#         train_transform=None,
#         val_transform=None,
#         split=(0.5, 0.5),
#         batch_size=32,
#         *args, **kwargs):
#     """
#     This function returns the train, val and test dataloaders.
#     """
#     # create the datasets
#     train_ds = ImageFolder(root=train_dir, transform=train_transform)
#     val_ds = ImageFolder(root=val_dir, transform=val_transform)
#     # now we want to split the val_ds in validation and test
#     lengths = np.array(split) * len(val_ds)
#     lengths = lengths.astype(int)
#     left = len(val_ds) - lengths.sum()
#     # we need to add the different due to float approx to int
#     lengths[-1] += left

#     val_ds, test_ds = random_split(val_ds, lengths.tolist())
#     logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
#     val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
#     test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

#     return train_dl, val_dl, test_dl


def get_dataloaders(
        train_dir,
        val_dir,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    
    # load train dataset
    train_ds = pd.read_pickle(train_dir / "dfTrain3.pkl")
    colsToExc = ['DATE', 'DeltaTimeTri', 'Lot']
    train_ds.drop(colsToExc, axis=1, inplace=True)
    tarCols = [x for x in train_ds.keys().to_list() if 'RsP' in x]  # extract targets
    dfFea = train_ds.drop(tarCols, axis=1)
    XTrain = np.array(dfFea)
    yTrain = np.array(train_ds[tarCols])
    
    # load val dataset
    val_ds = pd.read_pickle(val_dir / "dfVal3.pkl")
    val_ds.drop(colsToExc, axis=1, inplace=True)
    dfFea = val_ds.drop(tarCols, axis=1)
    XVal = np.array(dfFea)
    yVal = np.array(val_ds[tarCols])
    
    # load test dataset
    test_ds = pd.read_pickle(val_dir / 'dfTest3.pkl')
    test_ds.drop(colsToExc, axis=1, inplace=True)
    dfFea = test_ds.drop(tarCols, axis=1)
    XTest = np.array(dfFea)
    yTest = np.array(test_ds[tarCols])
    
    # build the datasets
    train_ds = CustomDataset(XTrain, yTrain)
    val_ds = CustomDataset(XVal, yVal)
    test_ds = CustomDataset(XTest, yTest)
    
    print()
    logging.info(f'Train samples = {len(train_ds)}, Validation samples = {len(val_ds)}, Test samples = {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl
