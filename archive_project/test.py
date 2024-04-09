import datasets

# Load the dataset from the saved directory
train_dataset = datasets.load_from_disk('./dataset/train_dataset/')

# Access the dataset
print(train_dataset[55])