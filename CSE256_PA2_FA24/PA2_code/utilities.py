import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size,encoder = True):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long,device=torch.device("cuda")).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)
        if encoder:
            # Process the input tensor through the encoder model
            _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps
        else:
            padded_sentence += [0]
            x = padded_sentence[:-1]
            y = padded_sentence[1:]
            x = torch.tensor(x , dtype= torch.long, device = torch.device("cuda")).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.long, device=torch.device("cuda")).unsqueeze(0)
            _, attn_maps = self.model(x,y)
        # Only check the first layer
        attn_maps = attn_maps[0]
        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            att_map = att_map[0]
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")
            
            # Show the plot
            plt.show()
            


