from common.core import Registry

Datasets = Registry("datasets")

def build_dataset(dataset_name, **kwargs):
    try:
        dataset = Datasets.get(dataset_name)
        return dataset(**kwargs)
    except Exception as error:
        print("dataset load error:", error)
        return None 
