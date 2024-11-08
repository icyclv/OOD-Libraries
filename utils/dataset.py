from torchvision import datasets, transforms
from utils.ImageFolderLMDB import ImageFolderLMDB
# import os

def get_dataset(dataset, mode="test"):

    train_dataset = None
    test_dataset = None

    # ind dataset

    # small-scale dataset
    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        size = 32
        if mode == "test":
            train_transform = transforms.Compose([
                transforms.Resize([size,size]), 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize([size,size]), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        test_transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = CIFAR10("./data/cifar10", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10("./data/cifar10", train=False, transform=test_transform, download=True)

    elif dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_dataset = CIFAR100("./data/cifar100", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100("./data/cifar100", train=False, transform=test_transform, download=True)
    
    # large-scale dataset
    elif dataset == "ImageNet":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = ImageFolderLMDB(db_path='./data/ImageNet-1000/imagenet/train.lmdb', transform=transform_test_largescale)
        test_dataset = ImageFolderLMDB(db_path='./data/ImageNet-1000/imagenet/val.lmdb', transform=transform_test_largescale)
        # train_dataset = datasets.ImageFolder(root='./data/ImageNet-1000/imagenet/train', transform=transform_test_largescale)
        # test_dataset = datasets.ImageFolder(root='./data/ImageNet-1000/imagenet/val', transform=transform_test_largescale)


    # ood dataset

    # small-scale dataset
    elif dataset == "svhn":
        from torchvision.datasets import SVHN
        size = 32
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = SVHN("./data/svhn", split='test', transform=transform, download=True)
    
    elif dataset == "dtd":
        size = 32
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/dtd/images', transform=transform)
    
    elif dataset == "places365":
        size = 32
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/Places', transform=transform)

    elif dataset == "iSUN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/iSUN', transform=transform)
        
    elif dataset == "LSUN_crop":
        transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/LSUN', transform=transform)

    elif dataset == "LSUN_resize":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/LSUN_resize', transform=transform)

    elif dataset == "TinyImageNet_crop":
        crop_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/TinyImagenet-crop', transform=crop_transform)

    elif dataset == "TinyImageNet_resize":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/TinyImagenet-resize', transform=transform)


    
    # large-scale dataset
    elif dataset == "iNat":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        # test_dataset = datasets.ImageFolder(root='./data/iNaturalist', transform=transform_test_largescale)
        test_dataset = ImageFolderLMDB(db_path='./data/iNaturalist.lmdb', transform=transform_test_largescale)


    elif dataset == "SUN":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/SUN.lmdb', transform=transform_test_largescale)


    elif dataset == "Places":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/Places.lmdb', transform=transform_test_largescale)

    elif dataset == "Textures":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='./data/dtd/images', transform=transform_test_largescale)

    elif dataset == "NINCO":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/ninco.lmdb', transform=transform_test_largescale)

    elif dataset == "SSB_hard":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/ssb_hard.lmdb', transform=transform_test_largescale)

    elif dataset == "OpenImage_O":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/openimage_o.lmdb', transform=transform_test_largescale)

    elif dataset == "ImageNet_O":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = ImageFolderLMDB(db_path='./data/imagenet-o.lmdb', transform=transform_test_largescale)

    return train_dataset, test_dataset