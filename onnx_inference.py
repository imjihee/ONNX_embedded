import pdb
import numpy as np

import onnx
import torch
import torchvision
import onnxruntime
import onnx.shape_inference as shape_inference

from PIL import Image
import torchvision.transforms as transforms

file_name = "resnet50_20ep.onnx"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

testset = torchvision.datasets.CIFAR10(root='/home/esoc/datasets/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
testiter = iter(testloader)

#pdb.set_trace()
#check_model: check if the loaded model has valid schema. check model version, graph architecture, nodes, input/outputs

"""
Load onnx model and inference
"""
model = onnx.load(file_name)
onnx.checker.check_model(model)

session = onnxruntime.InferenceSession(file_name)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

correct = 0.0
total = 0.0

while True:
    try:
        batch = next(testiter)
    except StopIteration:
        break
    x = batch[0]
    labels = np.array(batch[1])
    ort_inputs = {session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = session.run(None, ort_inputs)

    predicted = np.argmax(ort_outs[0], axis=1)
    correct += (predicted == labels).sum().item()
    total += len(labels)
    accuracy = (100 * correct / total)

print('Accuracy: %.2f' % (accuracy))     
