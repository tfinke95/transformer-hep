import torch



model_path='../../test_results/Part_pt_1/TTBar_run_test__part_pt_const128_403030_3_O0KHIRP/model_best.pt'
# Load the saved model
model = torch.load(model_path,map_location=torch.device('cpu'))

print(model)

transformer_layers = []


exit()
for name, module in model.named_children():
    if isinstance(module, torch.nn.TransformerEncoderLayer):
        transformer_layers.append((name, module))
    elif isinstance(module, torch.nn.Transformer):
        transformer_layers.append((name, module.encoder))

# Now, transformer_layers contains the Transformer layers
for name, layer in transformer_layers:
    print(f"Layer name: {name}, Layer type: {type(layer)}")
