import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyLSTM

torch.manual_seed(0)

PATH_TRAIN_SEQS = "../data/sepsis/processed/sepsis.seqs.train"
PATH_TRAIN_LABELS = "../data/sepsis/processed/sepsis.labels.train"
PATH_VALID_SEQS = "../data/sepsis/processed/sepsis.seqs.validation"
PATH_VALID_LABELS = "../data/sepsis/processed/sepsis.labels.validation"
PATH_TEST_SEQS = "../data/sepsis/processed/sepsis.seqs.test"
PATH_TEST_LABELS = "../data/sepsis/processed/sepsis.labels.test"
PATH_OUTPUT = "../out/result/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 0

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

num_features = calculate_num_features(train_seqs)

train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
                         num_workers=NUM_WORKERS)

model = MyLSTM(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

device = torch.device("cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy
        torch.save(model, os.path.join(PATH_OUTPUT, "MyLSTM.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyLSTM.pth"))


plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
valid_loss, valid_accuracy, valid_results = evaluate(best_model, device, valid_loader, criterion)

class_names = ['NO', 'YES']
plot_confusion_matrix(valid_results, class_names)


def predict_sepsis(model, device, data_loader):
    probas = []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)

            output = torch.softmax(model(input), 1)

            y_pred = output.detach().to('cpu').select(1, 1).numpy().tolist()

            probas.extend(y_pred)

    return probas


test_prob = predict_mortality(best_model, device, test_loader)
test_id = pickle.load(open(PATH_TEST_IDS, "rb"))