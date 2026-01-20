import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from RNN import RNN_base, RNN_lstm, RNN_gru
from engine import DataLoader, Training, generate_text

def data_prep():
    ds_text = ''
    with open('/home/mehdi/DL-from-scratch/src/RNN/tinyshakespeare.txt', 'r') as f:
        ds_text = f.read()
    ds_text = ds_text.lower() # convert to lowercase to make training easier

    vocab = sorted(set(ds_text))
    char_to_idx = {char:idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx:char for char, idx in char_to_idx.items()}

    corpus_indices = torch.tensor([char_to_idx[char] for char in ds_text])
    # Optimization: encoded_text should be on CPU initially if memory is an issue, 
    # but here it's small enough to handle.
    encoded_text = nn.functional.one_hot(corpus_indices).float()
    
    return corpus_indices, encoded_text, vocab, char_to_idx, idx_to_char

def benchmark_models(corpus_indices, encoded_text, vocab, char_to_idx, idx_to_char, device, epochs=5):
    results = {}
    num_hiddens = 512
    seq_length = 64
    batch_size = 128
    lr = 0.5
    
    data_loader = DataLoader(corpus_indices, encoded_text, seq_length, batch_size)
    
    models_config = [
        ("Vanilla RNN", RNN_base, [nn.Tanh()]),
        ("LSTM", RNN_lstm, []),
        ("GRU", RNN_gru, [])
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, model_class, extra_args in models_config:
        print(f"\n{'='*20}\nBenchmarking {name}\n{'='*20}")
        
        # Instantiate model with output size as vocab size
        model = model_class(len(vocab), num_hiddens, len(vocab), *extra_args)
        model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        trainer = Training(model, data_loader, optimizer, epochs, vocab, char_to_idx, idx_to_char)
        
        start_time = time.time()
        losses = trainer.train()
        end_time = time.time()
        
        time_taken = end_time - start_time
        print(f"Time taken for {name}: {time_taken:.2f} seconds")
        
        results[name] = {
            'time': time_taken,
            'losses': losses
        }

        
        plt.plot(range(1, epochs + 1), losses, label=f"{name} ({time_taken:.1f}s)")
        
        # Final Sample
        print(f"\nFinal {name} sample:")
        print(generate_text(model, 'the ', 50, vocab, char_to_idx, idx_to_char))
        print("-" * 50)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Architectures Comparison on Shakespeare')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_benchmark.png')
    print("\nBenchmark plot saved as 'rnn_benchmark.png'")
    
    # Print summary table
    print("\n" + "="*50)
    print(f"{'Model':<15} | {'Best Loss':<12} | {'Time (s)':<10}")
    print("-" * 43)
    for name, data in results.items():
        best_loss = min(data['losses'])
        time_taken = data['time']
        print(f"{name:<15} | {best_loss:<12.4f} | {time_taken:<10.2f}")
    print("="*50 + "\n")
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    corpus_indices, encoded_text, vocab, char_to_idx, idx_to_char = data_prep()
    
    benchmark_models(corpus_indices, encoded_text, vocab, char_to_idx, idx_to_char, device, epochs=10)

if __name__ == "__main__":
    main()
