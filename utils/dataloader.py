from torch.utils.data import DataLoader
from .dataset import get_datasets, get_extract_datasets, get_real_datasets, get_attr_datasets

def get_dataloaders(args):
    train_ds, valid_ds, test_ds = get_datasets(args)
    print('Train Dataset Size', len(train_ds))
    print('Valid Dataset Size', len(valid_ds))
    print('Test Dataset Size', len(test_ds))
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_ds, args.batch_size, False, num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=args.num_workers)
    return train_dl, valid_dl, test_dl

def get_extract_dataloaders(args):
    train_ds, valid_ds, test_ds = get_extract_datasets(args)
    print('Train Dataset Size', len(train_ds))
    print('Valid Dataset Size', len(valid_ds))
    print('Test Dataset Size', len(test_ds))
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, 
                          collate_fn=train_ds.collate_fn)
    valid_dl = DataLoader(valid_ds, args.batch_size, False, num_workers=args.num_workers, 
                          collate_fn=valid_ds.collate_fn)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=args.num_workers, 
                         collate_fn=test_ds.collate_fn)
    return train_dl, valid_dl, test_dl

def get_real_dataloaders(args):
    train_ds, valid_ds, test_ds = get_real_datasets(args)
    print('Train Dataset Size', len(train_ds))
    print('Valid Dataset Size', len(valid_ds))
    print('Test Dataset Size', len(test_ds))
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, 
                          collate_fn=train_ds.collate_fn)
    valid_dl = DataLoader(valid_ds, args.batch_size, False, num_workers=args.num_workers, 
                          collate_fn=valid_ds.collate_fn)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=args.num_workers, 
                         collate_fn=test_ds.collate_fn)
    return train_dl, valid_dl, test_dl

def get_attr_dataloaders(args):
    train_ds, valid_ds, test_ds, num_labels = get_attr_datasets(args)
    print('Train Dataset Size', len(train_ds))
    print('Valid Dataset Size', len(valid_ds))
    print('Test Dataset Size', len(test_ds))
    print('Attribute Label Num', num_labels)
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, 
                          collate_fn=train_ds.collate_fn)
    valid_dl = DataLoader(valid_ds, args.batch_size, False, num_workers=args.num_workers, 
                          collate_fn=valid_ds.collate_fn)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=args.num_workers, 
                         collate_fn=test_ds.collate_fn)
    return train_dl, valid_dl, test_dl, num_labels