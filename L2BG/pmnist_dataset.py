from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np 

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )

def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get_dataset(name, train=True, download=True, permutation=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    return dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )

def get_Pmnist_tasks(n_tasks,batch_size,cuda=False):
    train_loader = {}
    val_loader = {}
    for i in range(n_tasks):
        p = np.random.permutation(DATASET_CONFIGS['mnist']['size']**2)
        train_dataset = get_dataset('mnist', permutation=p)
        val_dataset = get_dataset('mnist', train=False, permutation=p)
        train_loader['task_'+str(i)] = get_data_loader(
                    train_dataset, batch_size=batch_size,
                    cuda=cuda
                )
        val_loader['task_'+str(i)] = get_data_loader(
                    val_dataset, batch_size=batch_size,
                    cuda=cuda
                )
    return train_loader, val_loader


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10}
}

