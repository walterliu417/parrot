import torch
import torch.onnx
from nn_creator import *

# configuring device
try:
    device = xm.xla_device()
    print("Running on the TPU")
except:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on the GPU')
        torch.cuda.synchronize()
    else:
        device = torch.device('cpu')
        print('Running on the CPU')

model = ChessEvaluationNet()

checkpoint_file = torch.load(r"D:\Parakeet\models\best_parakeet_1.pickle", weights_only=True, map_location=device)
# print(checkpoint_file)
model.load_state_dict(checkpoint_file)
model.to(device=device)
model.eval()

# Input to the model
x = torch.randn(64, 2, 8, 8, requires_grad=True, device=device)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "C:/Projects/Parakeet/best_parakeet_3.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=18,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes


              'output' : {0 : 'batch_size'}})
