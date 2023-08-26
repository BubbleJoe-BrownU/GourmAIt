from torchvision.datasets import Food101

print("Start to download Food 101 dataset, might take ~ minutes")
Food101(root="datasets", download=True)
