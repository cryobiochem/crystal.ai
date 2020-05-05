import os
selected_model_dir = "GPU Model"

model_path = os.listdir('model/'+selected_model_dir)
for model_file in model_path:
    print(model_file)
    print(model_file[-2:])
    if model_file[-2:] == 'h5':
        model = model_file
print(model)
