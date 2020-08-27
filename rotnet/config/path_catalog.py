import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'FashionMNIST': {
            'data_dir': ''
        },
        'CIFAR10': {
            'data_dir': ''
        },
        'CIFAR100': {
            'data_dir': ''
        }
    }

    @staticmethod
    def get(name):
        if "FashionMNIST".__eq__(name):
            fashion_mnist_root = DatasetCatalog.DATA_DIR
            if 'MNIST_ROOT' in os.environ:
                fashion_mnist_root = os.environ['MNIST_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(fashion_mnist_root, attrs["data_dir"]),
            )
            return dict(factory="FashionMNIST", args=args)
        elif 'CIFAR100'.__eq__(name):
            # 先cifar100再cifar10
            cifar100_root = DatasetCatalog.DATA_DIR
            if 'CIFAR_ROOT' in os.environ:
                cifar100_root = os.environ['CIFAR_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(cifar100_root, attrs["data_dir"]),
            )
            return dict(factory="CIFAR100", args=args)
        elif 'CIFAR10'.__eq__(name):
            cifar10_root = DatasetCatalog.DATA_DIR
            if 'CIFAR_ROOT' in os.environ:
                cifar10_root = os.environ['CIFAR_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(cifar10_root, attrs["data_dir"]),
            )
            return dict(factory="CIFAR10", args=args)
        else:
            raise RuntimeError("Dataset not available: {}".format(name))
