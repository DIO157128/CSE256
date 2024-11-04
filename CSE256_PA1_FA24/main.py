# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer

from DANmodels import SentimentDatasetDAN, DAN, SentimentDatasetDANBPE, DANBPE
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

    return all_train_accuracy, all_test_accuracy

def trainBPE(vocab_size):
    texts = []
    lines = open('./data/train.txt','r').read().splitlines()
    for l in lines:
        texts.append(l.split('\t')[1])
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["PAD", "UNK"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save("./data/bpe_tokenizer_{}.json".format(vocab_size))


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader,
                                                           test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader,
                                                           test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_BOW.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_BOW.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load dataset
        start_time = time.time()
        indexer = read_word_embeddings('./data/glove.6B.50d-relativized.txt').word_indexer
        train_data = SentimentDatasetDAN("data/train.txt", indexer)
        dev_data = SentimentDatasetDAN("data/dev.txt", indexer)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"data load finished in : {elapsed_time} seconds")

        # Start training using 50d embedding
        print('\nglove.6B.50d:')
        start_time = time.time()
        embedding = read_word_embeddings('./data/glove.6B.50d-relativized.txt')
        vocab_size = len(embedding.vectors)
        glove50d_train_accuracy, glove50d_test_accuracy = experiment(
            DAN(embedding=embedding, hidden_size=100, output_size=2), train_loader,
            test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DAN using glove.6b.50d as embedding to train finished in : {elapsed_time} seconds")

        # Start training using 300d embedding
        print('\nglove.6B.300d:')
        start_time = time.time()
        embedding = read_word_embeddings('./data/glove.6B.300d-relativized.txt')
        glove300d_train_accuracy, glove300d_test_accuracy = experiment(
            DAN(embedding=embedding, hidden_size=100, output_size=2), train_loader,
            test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DAN using glove.6b.300d as embedding to train finished in : {elapsed_time} seconds")

        # Start training using random50d embedding
        print('\nrandom.50d:')
        start_time = time.time()
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=50)
        nn.init.uniform_(embedding.weight, a=-0.1, b=0.1)
        random50d_train_accuracy, random50d_test_accuracy = experiment(
            DAN(embedding=embedding, hidden_size=100, output_size=2, random=True), train_loader,
            test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DAN using random.50d as embedding to train finished in : {elapsed_time} seconds")

        # Start training using random50d embedding
        print('\nrandom.300d:')
        start_time = time.time()
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        nn.init.uniform_(embedding.weight, a=-0.1, b=0.1)
        random300d_train_accuracy, random300d_test_accuracy = experiment(
            DAN(embedding=embedding, hidden_size=100, output_size=2, random=True), train_loader,
            test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DAN using random.300d as embedding to train finished in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(glove50d_train_accuracy, label='glove50d')
        plt.plot(glove300d_train_accuracy, label='glove300d')
        plt.plot(random50d_train_accuracy, label='random50d')
        plt.plot(random300d_train_accuracy, label='random300d')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for Glove and Random Embeddings')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_DAN.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(glove50d_test_accuracy, label='glove50d')
        plt.plot(glove300d_test_accuracy, label='glove300d')
        plt.plot(random50d_test_accuracy, label='random50d')
        plt.plot(random300d_test_accuracy, label='random300d')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for Glove and Random Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_DAN.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

    elif args.model == "SUBWORDDAN":
        vocab_sizes = [2000,4000,6000,8000,10000,12000,14000]
        train_accuracys,test_accuracys = [],[]
        for vocab_size in vocab_sizes:
            print("Current vocab size: {}".format(vocab_size))
            # train BPE
            start_time = time.time()
            trainBPE(vocab_size)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"BPE training with vocab size of {vocab_size} finished in : {elapsed_time} seconds")

            # load data
            start_time = time.time()
            tokenizer = Tokenizer.from_file("./data/bpe_tokenizer_{}.json".format(vocab_size))
            train_data = SentimentDatasetDANBPE("data/train.txt", tokenizer)
            dev_data = SentimentDatasetDANBPE("data/dev.txt", tokenizer)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"data load finished in : {elapsed_time} seconds")

            print('\nvocab_size{}:'.format(vocab_size))
            start_time = time.time()
            cur_train_accuracy, cur_test_accuracy = experiment(
                DANBPE(vocab_size=vocab_size, hidden_size=100, output_size=2), train_loader,
                test_loader)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"DAN using BPE vocab size of {vocab_size} as embedding to train finished in : {elapsed_time} seconds")
            train_accuracys.append(cur_train_accuracy)
            test_accuracys.append(cur_test_accuracy)
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        for vocab_size,cur_train_accuracy in zip(vocab_sizes,train_accuracys):
            plt.plot(cur_train_accuracy, label=vocab_size)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for BPE with different vocab size')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_BPE.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        for vocab_size,cur_test_accuracy in zip(vocab_sizes,test_accuracys):
            plt.plot(cur_test_accuracy, label=vocab_size)
        plt.xlabel('Epochs')
        plt.ylabel('Testing Accuracy')
        plt.title('Testing Accuracy for BPE with different vocab size')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_BPE.png'
        plt.savefig(testing_accuracy_file)
        print(f"\n\nTesting accuracy plot saved as {testing_accuracy_file}")
if __name__ == "__main__":
    main()