"""Build the dataset."""

from dataset.reds import REDS


def build_dataset(dataset_config):
    """Build a dataset and make `DataLoader`.

     A sample `dataset_config` may look like the following
        dataset_config = {
            'dataloader_settings': {
                'train': {
                    'batch_size': 16,
                    'drop_remainder': True,
                    'shuffle': True
                },
                'val': {
                    'batch_size': 1
                }
            }
        }

    Args:
        dataset_config: A `dict` contains information to create a dataset.

    Returns:
        A dataset `dict` contains dataloader for different split.
    
    Raises:
        ValueError: If `split` is not supported in the dataset.
    """
    dataloader = {}

    dataloader_settings = [('train', {
        'batch_size': 16,
        'drop_remainder': True,
        'shuffle': False
    }), ('val', {
        'batch_size': 1
    })]
    # create datasets and dataloaders with different splits.
    for split, dataloader_setting in dataloader_settings:
        # check the given split is valid or not.

        # build dataset
        dataset = REDS(dataset_config, split=split.lower())
        dataset = dataset.build(**dataloader_setting)

        dataloader[split.lower()] = dataset

    return dataloader
