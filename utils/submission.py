
from torch.utils.data import DataLoader

import pathlib
import sys
import os
import torch
module_dir = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(module_dir)

import data_utils
import config as cf
from w4c_dataloader import create_dataset
from benchmarks.SmatUnet import FeaturesSysUNet as Model

# ------------
# 1. create folder structures for the submission
# ------------
def create_directory_structure(root, folder_name='submission'):
    """
    create competition output directory structure at given root path. 
    """
    challenges = {'w4c-core-stage-1': ['R1', 'R2', 'R3'], 'w4c-transfer-learning-stage-1': ['R4', 'R5', 'R6']}
    
    for f_name, regions in challenges.items():
        for region in regions:
            r_path = os.path.join(root, folder_name, f_name, region, 'test')
            try:
                os.makedirs(r_path)
                print(f'created path: {r_path}')
            except:
                print(f'failed to create directory structure, maybe they already exist: {r_path}')

# ------------
# 2. load data & model
# ------------
def get_data_iterator(region_id='R1', data_split= 'test', collapse_time=True, 
                      batch_size=1, shuffle=False, num_workers=0):
    """ creates an iterator for data in region 'region_id' for the 'data_split' data partition """
    
    params = cf.get_params(region_id=region_id)
    params['data_params']['collapse_time'] = collapse_time

    ds = create_dataset(data_split, params['data_params'])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    data_splits, test_sequences = data_utils.read_splits(params['data_params']['train_splits'], params['data_params']['test_splits'])
    test_dates = data_splits[data_splits.split=='test'].id_date.sort_values().values

    return iter(dataloader), test_dates, params

def load_model(Model, params, checkpoint_path='', device=None):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    
    if checkpoint_path == '':
        model = Model(params['model_params'], **params['data_params'])            
    else:
        print("model:", Model)
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path)
        
    if device is not None:
        model = model.eval().cuda(device)
        
    return model

# ------------
# 3. make predictions & loop over regions
# ------------
def get_preds(model, batch, device=None):
    """ computes the output of the model on the next iterator's batch 
        returns the prediction and the date of it
    """
    
    in_seq, out, metadata = batch
    day_in_year = metadata['in']['day_in_year'][0][0].item()
    
    if device is not None:
        in_seq = in_seq.cuda(device=device)
    y_hat = model(in_seq)
    y_hat = torch.squeeze(y_hat)
    y_hat = y_hat.data.cpu().numpy()  
    
    return y_hat, day_in_year

def predictions_per_day(test_dates, model, ds_iterator, device, file_path, data_params):
    """ computes predictions of all dates and saves them to disk """

    for target_date in test_dates:
        
        print(f'generating submission for date: {target_date}...')
        batch = next(ds_iterator)
        with torch.no_grad():
            y_hat, predicted_day = get_preds(model, batch, device)
        # force data to be in the valid range
        y_hat = y_hat.reshape(32,4,256,256)
        y_hat[y_hat>1] = 1
        y_hat[y_hat<0] = 0
        
        # batches are sorted by date for the dataloader, that's why they coincide
        assert predicted_day==target_date, f"Error, the loaded date {predicted_day} is different than the target: {target_date}"

        f_path = os.path.join(file_path, f'{predicted_day}.h5')
        y_hat = data_utils.postprocess_fn(y_hat, data_params['target_vars'], data_params['preprocess']['source'])
        data_utils.write_data(y_hat, f_path)
        print(f'--> saved in: {f_path}')

def main():
        # 1. Define model's checkpoints, regions per task & gpu id to use
    root_to_ckps = ''
    checkpoint_paths = {'R1': f'{root_to_ckps}R3.ckpt', 
                        'R2': f'{root_to_ckps}R3.ckpt', 
                        'R3': f'{root_to_ckps}R3.ckpt'}
    challenges = {'w4c-core-stage-1': ['R1', 'R2', 'R3'], 'w4c-transfer-learning-stage-1': ['R4', 'R5', 'R6']}
    device = 0 # gpu id - SET THE ID OF THE GPU YOU WANT TO USE

    # 2. define root and name of the submission to create the folders' structure
    root = 'submission_examples/UNet'
    folder_name = 'UNet_submission'
    create_directory_structure(root, folder_name=folder_name)
    
    task_name = 'w4c-core-stage-1'
    for region in challenges[task_name]:
        # load data and model
        
        ds_iterator, test_dates, params = get_data_iterator(region_id=region)
        model = load_model(Model, params, checkpoint_path=checkpoint_paths[region], device=device)

        r_path = os.path.join(root, folder_name, task_name, region, 'test')
        predictions_per_day(test_dates, model, ds_iterator, device, r_path, params['data_params'])

if __name__ == "__main__":
    main()
