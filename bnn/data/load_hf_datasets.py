from torch import stack, tensor
from datasets import load_dataset


def collate_fn(examples):
    pixel_values = stack([example["pixel_values"] for example in examples])
    labels = tensor([example["label"] for example in examples])
    return pixel_values, labels
    # return {"pixel_values": pixel_values, "labels": labels}


def get_ds(dataset_name, data_path, num_proc=4):
    dataset = load_dataset(dataset_name,
                        #    split='train[:50000]',
                           num_proc=num_proc,
                        #    cache_dir='../../dataset_caches/imagenet-1k',
                           cache_dir=data_path,
                           token='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'
                           )
    # dataset = dataset.train_test_split(0.2, )
    test_name = 'validation'
    # test_name = 'test'
    print(dataset)

    # train_len = len(dataset['train'])
    # test_len = len(dataset[test_name])

    # labels, label2id, id2label = get_labels(dataset)
    # split up training into training + validation
    train_ds = dataset['train']
    test_ds = dataset[test_name]
    # val_ds = dataset['validation']

    return train_ds, test_ds


def get_labels(a_dataset):
    labels = a_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    # id2label[2]
    return labels, label2id, id2label


def data_preprocess(example_batch, transform_f):
    example_batch["pixel_values"] = [
        transform_f(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch
