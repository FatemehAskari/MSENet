import os
import shutil
import time
import pprint
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from torchvision.utils import save_image

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2



def evaluate_on_one_task_test(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        model
    ) -> [int, int]:
        predictions = torch.max(
                        model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                        .detach()
                        .data,
                        1,
                    )[1]
        correct_predictions = (predictions == query_labels.cuda()).sum().item()
        total_predictions = len(query_labels)
        
        # Save misclassified images
        for i in range(len(query_labels)):
            if predictions[i] != query_labels[i]:
                # support_img = support_images[i].cpu()
                query_img = query_images[i].cpu()
                true_label = query_labels[i].item()
                predicted_label = predictions[i].item()
                
                # Create a transform to convert the tensor to an image
                transform = transforms.ToPILImage()
                
                # Convert the tensors to images
                # support_img = transform(support_img)
                query_img = transform(query_img)
                print(query_img.size())
                # Save support image with true label
                # support_filename = f'misclassified/support_{i}_true_{true_label}.png'
                # save_image(support_img, support_filename)
                
                # Save query image with predicted label
                # query_filename = f'misclassified/query_{i}_predicted_{predicted_label}.png'
                # save_image(query_img, query_filename)
        
        return correct_predictions, total_predictions
    
    
def evaluate_test(data_loader: DataLoader,model):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task_test(
                support_images, support_labels, query_images, query_labels,model
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )
    return  (100 * correct_predictions/total_predictions)   


def evaluate_on_one_task(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        model
    ) -> [int, int]:
        """
        Returns the number of correct predictions of query labels, and the total number of predictions.
        """
        

        return (
            torch.max(
                model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                .detach()
                .data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)




def evaluate(data_loader: DataLoader,model):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels,model
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )
    return  (100 * correct_predictions/total_predictions)   