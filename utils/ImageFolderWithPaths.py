# This is modified from Andrew Jong's Git Gist
# url: https://gist.githubusercontent.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d/raw/df4746fa46c3a06f5c041cec18a7eb66fb801197/pytorch_image_folder_with_file_paths.py

import torchvision.datasets as datasets



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# # Test code
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#
# data_dir = "../data/dataset/test_crop"
# # dataset = ImageFolderWithPaths(data_dir)  # our custom dataset
# # dataloader = torch.utils.data.DataLoader(dataset)
#
# dataloader = torch.utils.data.DataLoader(
#     ImageFolderWithPaths(data_dir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=256, shuffle=False, pin_memory=True)
#
# # iterate over data
# for inputs, labels, paths in dataloader:
#     # use the above variables freely
#     print(inputs, labels, paths)
