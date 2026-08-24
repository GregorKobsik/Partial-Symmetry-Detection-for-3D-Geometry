import json
import os.path as osp

from glob import glob

from torch_geometric.data import Dataset
from torch_geometric.io.obj import read_obj


class PartNet(Dataset):
    r"""The ModelNet10/40 datasets from the `"PartNet: A Large-scale Benchmark
    for Fine-grained and Hierarchical Part-level 3D Object Understanding"
    <https://partnet.cs.stanford.edu/>`_ paper,
    containing over 26.000 3D models cosisting of more than 573.000 annotated part instances.

    .. note::

        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Bag"`, :obj:`"Bed"`,
            :obj:`"Bottle"`, :obj:`"Bowl"`, :obj:`"Chair"`, :obj:`"Clock"`,
            :obj:`"Dishwasher"`, :obj:`"Display"`, :obj:`"Door"`, :obj:`"Earphone"`,
            :obj:`"Faucet"`, :obj:`"Hat"`, :obj:`"Knife"`, :obj:`"Lamp"`,
            :obj:`"Microwave"`, :obj:`"Mug"`, :obj:`"Refrigator"`, :obj:`"Scissors"`,
            :obj:`"StorageFurniture"`, :obj:`"Table"`, :obj:`"TrashCan"`, :obj:`"Vase"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        split (string, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): Unused. (default: :obj:`None`)
        pre_filter (callable, optional): Unused. (default: :obj:`None`)
    """

    categories = [
        'Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet', 'Hat',
        'Keyboard', 'Knife', 'Lamp', 'Microwave', 'Mug', 'Refrigerator', 'Scissors', 'StorageFurniture', 'Table',
        'TrashCan', 'Vase'
    ]
    splits = {'train': 0, 'val': 1, 'test': 2}
    stats_dir = 'metadata/stats/train_val_test_split'
    data_dir = 'data_v0'

    def __init__(
        self,
        root='data/PartNet',
        categories=None,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        if categories in (None, 'all', 'All', ['all'], ['All']):
            categories = list(self.categories)
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.categories for category in categories)
        self.categories = categories

        if split in self.splits:
            self.split = split
        else:
            raise ValueError((f'Split {split} found, but expected either train, val or test'))

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self) -> str:
        return osp.join('data/PartNet')

    @property
    def processed_file_names(self) -> str:
        return 'unprocessed.pt'

    def process(self):
        category_path = osp.join(self.root, self.stats_dir)
        self.data_file_names = []
        for category in self.categories:
            with open(osp.join(category_path, f"{category}.{self.split}.json")) as f:
                model_anno_ids = json.load(f)

            for model in model_anno_ids:
                anno_id = model["anno_id"]
                self.data_file_names += sorted(glob(osp.join(self.root, self.data_dir, f"{anno_id}/objs/*.obj")))

    def len(self):
        return len(self.data_file_names)

    def get(self, idx):
        return read_obj(self.data_file_names[idx])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'categories={self.categories})')
