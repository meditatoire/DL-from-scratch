import torch
import os
from transformers import Transformer
from engine import Tokenizer, TransformerDataLoader, DataManager, Seq2SeqEngine
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "fra.txt")
    src_texts, tgt_texts = DataManager.load_data(data_path)
    print(f"Loaded {len(src_texts)} pairs")
    
    # Build vocabulary
    full_text = src_texts + tgt_texts
    vocab_text = " ".join(full_text)
    tokenizer = Tokenizer(vocab_text, False)
    char_to_idx, idx_to_char = tokenizer.build_vocab()
    print(f"Vocabulary size: {len(char_to_idx)}")

    # Setup DataLoader
    train_loader = TransformerDataLoader(src_texts, tgt_texts, char_to_idx, max_len=30, batch_size=256)

    # Initialize Model
    model = Transformer(
        vocab_size=len(char_to_idx),
        num_hidden=128,
        num_heads=4,
        ffn_num_hidden=256,
        num_layers=2,
        dropout=0.1
    )

    # Initialize Engine and Train
    engine = Seq2SeqEngine(model, char_to_idx, device)
    
    num_epochs = 20
    print("Starting training...")
    loss_history = engine.train(train_loader, lr=0.001, num_epochs=num_epochs, char_to_idx=char_to_idx)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_graph.png")
    print("Saved loss graph to loss_graph.png")

    # Prediction
    test_sentence = "He is sleeping" 
    translation = engine.predict(test_sentence, char_to_idx, idx_to_char, 20)
    print(f"Input: {test_sentence}")
    print(f"Prediction: {translation}")

if __name__ == "__main__":
    main()
