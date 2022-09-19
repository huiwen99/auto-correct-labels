import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd

from .model import Classifier


def train_epoch(model, device, train_ld, val_ld, optimizer, criterion, epoch):
    """
    Trains model on training data for 1 epoch
    """
    model.train()
    with tqdm(train_ld, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description("Epoch {}".format(epoch))
            data, target = data.to(device), target.to(device)
            
            target = target.reshape(target.size(0)).long()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    train_loss, train_acc = evaluate(model, device, criterion, train_ld)
    print('Train Epoch: {} @ {} \nTrain Loss: {:.4f} - Train Accuracy: {:.1f}%'.format(epoch, datetime.datetime.now().time(), train_loss, train_acc))
    
    val_loss, val_acc = evaluate(model, device, criterion, val_ld)
    print('Val Loss: {:.4f} - Val Accuracy: {:.1f}%'.format(val_loss, val_acc))

    return train_loss, train_acc, val_loss, val_acc

def train_model(model_name, num_class, device, train_ld, val_ld, learning_rate, num_epochs):
    """
    Trains 1 model
    """
    model = Classifier(num_class, model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    temp_model_path = 'temp/temp_model.pt'
    patience = 3
    
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(1, num_epochs+1):
        _, _, _, val_acc = train_epoch(
            model, device, 
            train_ld, val_ld, 
            optimizer, criterion, 
            epoch
        )
        
        if val_acc >= best_val_acc:
            torch.save(model.state_dict(), temp_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                break
    model.load_state_dict(torch.load(temp_model_path))
    return model

def evaluate(model, device, criterion, data_ld):
    """
    Evalutes model and returns loss and accuracy
    """
    model.eval()
    loss = 0 
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in data_ld:
            data, target = data.to(device), target.to(device)
            target = target.reshape(target.size(0)).long()
            
            output = model(data)
            loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=False)
            
            correct += torch.eq(target, pred).sum().item()
            total_num += len(target)
            
    loss /= len(data_ld)
    acc = 100. * correct / total_num
    return loss, acc


def get_predictions(model, device, data_ld):
    """
    Get predictions from dataloader
    """
    predictions = []
    model.eval()
    with torch.no_grad():
        for data, _ in data_ld:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


def store_predictions(pred_tracker, dataset, predictions):
    """Store predictions in dictionary"""
    for i in range(len(dataset)):
        filename = dataset.x[i]
        label = dataset.y[i]
        corrected_label = dataset.reverse_class_map[predictions[i]]
        if filename in pred_tracker:
            pred_tracker[filename]['corrected_label'].append(corrected_label)
        else:
            pred_tracker[filename] = {
                'label': label,
                'corrected_label': [corrected_label]
            }
    return pred_tracker


def write_results(path, pred_tracker):
    """
    Write results into a csv file
    """
    df = {}
    for filename in pred_tracker:
        label = pred_tracker[filename]['label']
        corrected_labels = pred_tracker[filename]['corrected_label']
        corrected_label = max(set(corrected_labels), key = corrected_labels.count)
        num_preds = len(corrected_labels)
        percentage = corrected_labels.count(corrected_label) / num_preds * 100
        
        df[filename] = [label, corrected_label, num_preds, percentage]
        
    df = pd.DataFrame.from_dict(
        df, orient='index',
        columns=['label', 'corrected_label', 'num_preds', '%']
    )
    df.to_csv(path)
        