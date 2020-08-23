import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'fashion-mnist': {
            'data_dir': ''
        },
    }

    @staticmethod
    def get(name):
        if "fashion" in name:
            mnist_root = DatasetCatalog.DATA_DIR
            if 'MNIST_ROOT' in os.environ:
                mnist_root = os.environ['MNIST_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(mnist_root, attrs["data_dir"]),
            )
            return dict(factory="FashionMNIST", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
