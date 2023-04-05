from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
import copy

def train_model(model,trainloader,loss_fn,optimizer,training_loss_list,epoch,device,scheduler = None,decay_epochs = None):
    model.train()
    #total training loss on this epoch
    training_loss = 0
    for batch,(x,y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device).long()
        h0,c0 = model.init_hidden(model.num_batches)
        out = model(x,h0,c0)
        loss = loss_fn(out,y)
        training_loss += loss.item()
        prev_loss = loss.item()
        #sets all the existing gradients to zero so no accumulation occurs
        optimizer.zero_grad()
        #back propagation
        loss.backward()
        optimizer.step()
        # if batch%100 == 0:
        #     print(f"Finished Batch {batch}")
    epoch_average_loss = training_loss/len(trainloader)
    training_loss_list.append(epoch_average_loss)
    
    if decay_epochs != None:
        if epoch in decay_epochs:
            scheduler.step()
            print(f"Decayed LR : {scheduler.get_last_lr()}")
        
    return training_loss_list,model
    
#Validation error is calculated to check if there is any plateauing of the 
#decrease in error so that learning rate decay can be applied
def calculate_validation_error(model,dataloader,loss_fn,val_loss_list,device):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch,(x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device).long()
            h0,c0 = model.init_hidden(10)
            out = model(x,h0,c0)
            # out = model(x)
            loss = loss_fn(out,y)
            validation_loss += loss.item()
    epoch_average_validation_loss = validation_loss/len(dataloader)
    val_loss_list.append(epoch_average_validation_loss)
    return val_loss_list
    
    
    
def check_accuracy(model,dataloader,device,check_precision = False):
    model.eval()
    correct_total = 0
    total_instances = 0
    with torch.no_grad():
        for batch,(x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to('cpu').long()
            h0,c0 = model.init_hidden(10)
            out = model(x,h0,c0)
            out = out.to('cpu')
            _,max_index = out.max(dim = 1)
            correct_total += (max_index == y).sum()
            total_instances += len(y)
    acc = float(correct_total)/total_instances
    return acc
    
def train_epochs(num_epochs,model,train_dataloader,validation_dataloader,optimizer,loss_fn,device,epoch_divisor,scheduler = None,decay_epochs = None):
    train_loss_list = list()
    val_loss_list = list()
    accuracy_list_training = list()
    accuracy_list_validation = list()
    loss_fn_val_error = copy.deepcopy(loss_fn)
    for epoch in range(num_epochs):
        train_loss_list,model = train_model(model,train_dataloader,loss_fn,optimizer,train_loss_list,epoch,device,scheduler,decay_epochs)
        val_loss_list = calculate_validation_error(model,validation_dataloader,loss_fn_val_error,val_loss_list,device)
        training_accuracy = check_accuracy(model,train_dataloader,device)
        validation_accuracy = check_accuracy(model,validation_dataloader,device)
        accuracy_list_training.append(training_accuracy)
        accuracy_list_validation.append(validation_accuracy)
        if len(train_loss_list) > 1:
            if (train_loss_list[-1] / train_loss_list[-2]) >= 3:
                print(f"Error shot up at epoch {epoch}")
                break
        if epoch%epoch_divisor == 0:
            print(f"Finished Epoch {epoch}")
    return train_loss_list,val_loss_list,model,accuracy_list_training,accuracy_list_validation
    