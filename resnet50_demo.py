import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()  # Set the model to evaluation mode

batch_size = 1
# Create a dummy input tensor (batch size 1, 3 channels, height 224, width 224)
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Perform any necessary pre-processing on the dummy input
# For ResNet50, usually, you need to normalize the input using the same mean and std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize])
dummy_input = preprocess(Image.fromarray((dummy_input.squeeze().numpy().transpose((1, 2, 0)) * 255).astype('uint8')))

# Add an extra batch dimension to the input
dummy_input = dummy_input.unsqueeze(0)

# Run the model on the dummy input
# with torch.no_grad():
#     output = resnet50(dummy_input)
#     print("origin output:", output[0][:10])

resnet50_optimized = torch.compile(resnet50)
resnet50_optimized_xpu = resnet50_optimized.to("xpu")
dummy_input_xpu = dummy_input.to("xpu")

print("batch size:   ", batch_size)
print("timer start")
start_time = time.time()
# Run the model on the dummy input
with torch.no_grad():
    for iter in range(4):
        output_xpu = resnet50_optimized_xpu(dummy_input_xpu)
        output = output_xpu.to("cpu")
        print("opt output:", output[0][:10])
end_time = time.time()
print("timer end")
print("Model Output:", output.argmax(dim=1))
print(f"time cost:{end_time - start_time:.4f} sec")