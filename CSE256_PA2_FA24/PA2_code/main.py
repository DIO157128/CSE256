import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerClassifier, TransformerDecoder
from utilities import Utilities
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])),
                                               "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y)  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--part', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--sanity_check', action='store_true', default=False)
    # Parse the command-line arguments
    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.part == 'part1':
        # Prepare the dataset for part 1
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

        # Initialize the encoder model and do sanity check

        encoder_model = TransformerClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            hidden_dim=n_hidden,
            num_classes=n_output,
            dropout=0.1
        ).to(device)
        total_params = sum(p.numel() for p in encoder_model.parameters())
        trainable_params = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
        print(f"Encoder Total Parameters: {total_params}")
        print(f"Encoder Trainable Parameters: {trainable_params}")
        if args.sanity_check:
            print("\nPerforming encoder's attention matrix sanity checks and visualization:")
            encoder_utilities = Utilities(tokenizer, encoder_model)
            sample_sentence = "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest."
            encoder_utilities.sanity_check(sample_sentence, block_size)
        accuracys = []
        # Train encoder model
        optimizer = optim.Adam(encoder_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs_CLS):
            total_loss = 0
            encoder_model.train()
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                outputs, _ = encoder_model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy = compute_classifier_accuracy(encoder_model, train_CLS_loader)
            accuracys.append(accuracy)
            print(f"Epoch {epoch + 1}/{epochs_CLS} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        test_accuracy = compute_classifier_accuracy(encoder_model, test_CLS_loader)
        print(f"Testing Accuracy: {test_accuracy:.2f}%")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(accuracys, label='Encoder Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for Transformer Encoder Classifier')
        plt.legend()
        plt.grid()
        for epoch, accuracy in enumerate(accuracys):
            plt.text(epoch, accuracy, f'{accuracy:.2f}', ha='center', va='bottom')
        file_name = 'part1.png'
        plt.savefig(file_name)
    if args.part == 'part2':
        # Prepare dataset for part 2

        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        f.close()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f1:
            obamatestText = f1.read()
        with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f2:
            hbushtestText = f2.read()
        with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f3:
            wbushtestText = f3.read()
        f1.close()
        f2.close()
        f3.close()
        test_obama_dataset = LanguageModelingDataset(tokenizer, obamatestText, block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)
        test_hbush_dataset = LanguageModelingDataset(tokenizer, hbushtestText, block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, wbushtestText, block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the decoder model and do sanity check

        decoder_model = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            dropout=0.1
        ).to(device)

        total_params = sum(p.numel() for p in decoder_model.parameters())
        trainable_params = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
        print(f"Decoder Total Parameters: {total_params}")
        print(f"Decoder Trainable Parameters: {trainable_params}")
        if args.sanity_check:
            print("\nPerforming decoder's attention matrix sanity checks and visualization:")
            decoder_utilities = Utilities(tokenizer, decoder_model)
            sample_sentence = "It is costly and politically difficult to continue this conflict."
            decoder_utilities.sanity_check(sample_sentence, block_size, False)
        # Train decoder model
        optimizer = optim.Adam(decoder_model.parameters(), lr=learning_rate)
        decoder_model.train()
        perplexitys = []
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i > max_iters:
                break

            xb, yb = xb.to(device), yb.to(device)
            loss, _ = decoder_model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
                perplexity = compute_perplexity(decoder_model, train_LM_loader)
                print(f"Perplexity on train set: {perplexity}")
                perplexitys.append(perplexity)
        perplexity = compute_perplexity(decoder_model, train_LM_loader)
        print(f"Final Perplexity on train set: {perplexity}")
        obama_perplexity = compute_perplexity(decoder_model, test_obama_loader, eval_iters=eval_iters)
        hbush_perplexity = compute_perplexity(decoder_model, test_hbush_loader, eval_iters=eval_iters)
        wbush_perplexity = compute_perplexity(decoder_model, test_wbush_loader, eval_iters=eval_iters)
        print(f"Perplexity on Obama test set: {obama_perplexity}")
        print(f"Perplexity on H. Bush test set: {hbush_perplexity}")
        print(f"Perplexity on W. Bush test set: {wbush_perplexity}")
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(perplexitys, label='Decoder Perplexity', marker='o')  # Add marker to show points
        plt.xlabel('Epochs')
        plt.ylabel('Training Perplexity')
        plt.title('Training Perplexity for Transformer Decoder Generator')
        plt.legend()
        plt.grid()

        # Annotate each point with its value
        for epoch, perplexity in enumerate(perplexitys):
            plt.text(epoch, perplexity, f'{perplexity:.2f}', ha='center', va='bottom')

        file_name = 'part2.png'
        plt.savefig(file_name)
    if args.part == 'part3':
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

        # Initialize the encoder model and do sanity check

        encoder_model = TransformerClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            hidden_dim=n_hidden,
            num_classes=n_output,
            dropout=0.1,
            postion=True
        ).to(device)
        pos_encoder_model = TransformerClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            hidden_dim=n_hidden,
            num_classes=n_output,
            dropout=0.1,
            postion=True
        ).to(device)
        total_params = sum(p.numel() for p in pos_encoder_model.parameters())
        trainable_params = sum(p.numel() for p in pos_encoder_model.parameters() if p.requires_grad)
        print(f"Encoder Total Parameters: {total_params}")
        print(f"Encoder Trainable Parameters: {trainable_params}")
        if args.sanity_check:
            print("\nPerforming encoder's attention matrix sanity checks and visualization:")
            encoder_utilities = Utilities(tokenizer, pos_encoder_model)
            sample_sentence = "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest."
            encoder_utilities.sanity_check(sample_sentence, block_size)
        encoder_accuracys = []
        # Train encoder model
        optimizer = optim.Adam(encoder_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs_CLS):
            total_loss = 0
            encoder_model.train()
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                outputs, _ = encoder_model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy = compute_classifier_accuracy(encoder_model, train_CLS_loader)
            encoder_accuracys.append(accuracy)
            print(f"Epoch {epoch + 1}/{epochs_CLS} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        test_accuracy = compute_classifier_accuracy(encoder_model, test_CLS_loader)
        print(f"Encoder Testing Accuracy: {test_accuracy:.2f}%")
        pos_encoder_accuracys = []
        # Train pos_encoder model
        optimizer = optim.Adam(pos_encoder_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs_CLS):
            total_loss = 0
            pos_encoder_model.train()
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                outputs, _ = pos_encoder_model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy = compute_classifier_accuracy(pos_encoder_model, train_CLS_loader)
            pos_encoder_accuracys.append(accuracy)
            print(f"Epoch {epoch + 1}/{epochs_CLS} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        test_accuracy = compute_classifier_accuracy(pos_encoder_model, test_CLS_loader)
        print(f"Encoder with Positional Encoding Testing Accuracy: {test_accuracy:.2f}%")
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(encoder_accuracys, label='Encoder Accuracy', marker='o')
        plt.plot(pos_encoder_accuracys, label='Encoder Accuracy with Positional Encoding', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for Transformer Encoder Classifier')
        plt.legend()
        plt.grid()

        # Annotate each point with its value for encoder_accuracys
        for epoch, accuracy in enumerate(encoder_accuracys):
            plt.text(epoch, accuracy, f'{accuracy:.2f}', ha='center', va='bottom', color='blue')

        # Annotate each point with its value for pos_encoder_accuracys
        for epoch, accuracy in enumerate(pos_encoder_accuracys):
            plt.text(epoch, accuracy, f'{accuracy:.2f}', ha='center', va='bottom', color='orange')

        file_name = 'part3_encoder.png'
        plt.savefig(file_name)

        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        f.close()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f1:
            obamatestText = f1.read()
        with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f2:
            hbushtestText = f2.read()
        with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f3:
            wbushtestText = f3.read()
        f1.close()
        f2.close()
        f3.close()
        test_obama_dataset = LanguageModelingDataset(tokenizer, obamatestText, block_size)
        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)
        test_hbush_dataset = LanguageModelingDataset(tokenizer, hbushtestText, block_size)
        test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, wbushtestText, block_size)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the decoder model and do sanity check

        decoder_model = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            dropout=0.1
        ).to(device)
        pos_decoder_model = TransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=n_embd,
            num_heads=n_head,
            num_layers=n_layer,
            feedforward_dim=n_hidden,
            max_seq_length=block_size,
            dropout=0.1,
            position=True
        ).to(device)
        total_params = sum(p.numel() for p in pos_decoder_model.parameters())
        trainable_params = sum(p.numel() for p in pos_decoder_model.parameters() if p.requires_grad)
        print(f"Decoder Total Parameters: {total_params}")
        print(f"Decoder Trainable Parameters: {trainable_params}")
        if args.sanity_check:
            print("\nPerforming decoder's attention matrix sanity checks and visualization:")
            decoder_utilities = Utilities(tokenizer, pos_decoder_model)
            sample_sentence = "It is costly and politically difficult to continue this conflict."
            decoder_utilities.sanity_check(sample_sentence, block_size, False)
        # Train decoder model
        optimizer = optim.Adam(decoder_model.parameters(), lr=learning_rate)
        decoder_model.train()
        perplexitys = []
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i > max_iters:
                break

            xb, yb = xb.to(device), yb.to(device)
            loss, _ = decoder_model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
                perplexity = compute_perplexity(decoder_model, train_LM_loader)
                print(f"Perplexity on train set: {perplexity}")
                perplexitys.append(perplexity)
        perplexity = compute_perplexity(decoder_model, train_LM_loader)
        print(f"Final Perplexity on train set: {perplexity}")
        obama_perplexity = compute_perplexity(decoder_model, test_obama_loader, eval_iters=eval_iters)
        hbush_perplexity = compute_perplexity(decoder_model, test_hbush_loader, eval_iters=eval_iters)
        wbush_perplexity = compute_perplexity(decoder_model, test_wbush_loader, eval_iters=eval_iters)
        print(f"Perplexity on Obama test set: {obama_perplexity}")
        print(f"Perplexity on H. Bush test set: {hbush_perplexity}")
        print(f"Perplexity on W. Bush test set: {wbush_perplexity}")

        # Train decoder model
        optimizer = optim.Adam(pos_decoder_model.parameters(), lr=learning_rate)
        pos_decoder_model.train()
        pos_perplexitys = []
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i > max_iters:
                break

            xb, yb = xb.to(device), yb.to(device)
            loss, _ = pos_decoder_model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
                perplexity = compute_perplexity(pos_decoder_model, train_LM_loader)
                print(f"Perplexity on train set: {perplexity}")
                pos_perplexitys.append(perplexity)
        perplexity = compute_perplexity(pos_decoder_model, train_LM_loader)
        print(f"Final Perplexity on train set: {perplexity}")
        obama_perplexity = compute_perplexity(pos_decoder_model, test_obama_loader, eval_iters=eval_iters)
        hbush_perplexity = compute_perplexity(pos_decoder_model, test_hbush_loader, eval_iters=eval_iters)
        wbush_perplexity = compute_perplexity(pos_decoder_model, test_wbush_loader, eval_iters=eval_iters)
        print(f"Perplexity on Obama test set: {obama_perplexity}")
        print(f"Perplexity on H. Bush test set: {hbush_perplexity}")
        print(f"Perplexity on W. Bush test set: {wbush_perplexity}")
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(perplexitys, label='Decoder Perplexity', marker='o')  # Add marker to show points
        plt.plot(pos_perplexitys, label='Decoder Perplexity with Positional Encoding', marker='o')  # Add marker to show points
        plt.xlabel('Epochs')
        plt.ylabel('Training Perplexity')
        plt.title('Training Perplexity for Transformer Decoder Generator')
        plt.legend()
        plt.grid()

        for epoch, perplexity in enumerate(perplexitys):
            plt.text(epoch, perplexity, f'{perplexity:.2f}', ha='center', va='bottom')

        # Add a small offset to pos_perplexitys annotations
        offset = 10  # Adjust this value as needed
        for epoch, perplexity in enumerate(pos_perplexitys):
            plt.text(epoch, perplexity + offset, f'{perplexity:.2f}', ha='center', va='bottom', color='orange')

        file_name = 'part3_decoder.png'
        plt.savefig(file_name)
if __name__ == "__main__":
    main()
