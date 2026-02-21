import torch
import torch.nn as nn
import math
import random
import time
from transformers import *

# We use a simple character level tokenization
class Tokenizer():
    def __init__(self, corpus, use_path:bool):
        self.corpus = corpus
        self.use_path = use_path

    def build_vocab(self, add_special_tokens=True):
        ds_text = ""
        if self.use_path:
            with open(f"{self.corpus}", "r") as f:
                ds_text = f.read()
        else:
            ds_text = self.corpus
        vocab = sorted(set(ds_text))
        char_to_idx = {c:i for i,c in enumerate(vocab)}
        if add_special_tokens:
            char_to_idx["<bos>"] = len(vocab) + 2
            char_to_idx["<eos>"] = len(vocab) + 3
        char_to_idx["<unk>"] = len(vocab)
        char_to_idx["<pad>"] = len(vocab) + 1
        idx_to_char = {i:c for c,i in char_to_idx.items()}
        return char_to_idx, idx_to_char

    def tokenize(self, text):
        encoded_text = []
        for c in text:
            if c in char_to_idx:
                encoded_text.append(char_to_idx[c])
            else:
                encoded_text.append(char_to_idx["<unk>"])
        return encoded_text


class TransformerDataLoader:
    def __init__(self, src_texts, tgt_texts, char_to_idx, max_len=20, batch_size=4, shuffle=True):
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Pre-tokenize pairs during init to save time
        self.data = []
        for src, tgt in zip(src_texts, tgt_texts):
            # Tokenize Source
            src_tokens = [self.char_to_idx["<bos>"]]
            for char in src:
                src_tokens.append(self.char_to_idx.get(char, self.char_to_idx["<unk>"]))
            src_tokens.append(self.char_to_idx["<eos>"])
            
            # Tokenize Target
            tgt_tokens = [self.char_to_idx["<bos>"]]
            for char in tgt:
                tgt_tokens.append(self.char_to_idx.get(char, self.char_to_idx["<unk>"]))
            tgt_tokens.append(self.char_to_idx["<eos>"])
            
            self.data.append((src_tokens, tgt_tokens))

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            
            # Prepare batch lists
            encoder_inputs = []
            decoder_inputs = []
            decoder_targets = []
            src_valid_lens = []
            
            for src_seq, tgt_seq in batch:
                # Process encoder input
                # Truncate if needed
                if len(src_seq) > self.max_len:
                    src_seq = src_seq[:self.max_len]
                
                src_valid_lens.append(len(src_seq))
                
                # Pad Source
                pad_len = self.max_len - len(src_seq)
                if pad_len > 0:
                    src_padded = src_seq + [self.char_to_idx["<pad>"]] * pad_len
                else:
                    src_padded = src_seq
                encoder_inputs.append(src_padded)
                
                # Process decoder input/target
                # We need offset by 1 for Teacher Forcing
                # Input:  <bos> A B C ...
                # Target: A B C <eos> ...
                
                if len(tgt_seq) > self.max_len + 1:
                    tgt_seq = tgt_seq[:self.max_len + 1]
                    
                dec_in = tgt_seq[:-1]
                dec_out = tgt_seq[1:]
                
                # Pad Target
                pad_len = self.max_len - len(dec_in)
                if pad_len > 0:
                    padding = [self.char_to_idx["<pad>"]] * pad_len
                    dec_in_padded = dec_in + padding
                    dec_out_padded = dec_out + padding
                else:
                    dec_in_padded = dec_in[:self.max_len]
                    dec_out_padded = dec_out[:self.max_len]
                    
                decoder_inputs.append(dec_in_padded)
                decoder_targets.append(dec_out_padded)
            
            # Yield batch of tensors
            yield (
                torch.tensor(encoder_inputs),
                torch.tensor(decoder_inputs), 
                torch.tensor(decoder_targets),
                torch.tensor(src_valid_lens),
            )

# For gpt models (decoder only)
class DecoderDataLoader:
    def __init__(self, texts, char_to_idx, max_len=50, batch_size=4, shuffle=True):
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []

        for text in texts:
            tokens = [self.char_to_idx["<bos>"]]
            for char in text:
                tokens.append(self.char_to_idx.get(char, self.char_to_idx["<unk>"]))
            tokens.append(self.char_to_idx["<eos>"])
            if len(tokens) > self.max_len + 1:
                tokens = tokens[:self.max_len +1]
            self.data.append(tokens)

    def __len__(self):
        return (len(self.data) + self.batch_size- 1) // self.batch_size
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            batch_tokens = self.data[i:i+self.batch_size]
            X_batch = []
            Y_batch = []
            valid_lens = []
            for seq in batch_tokens:
                # Create X input and Y target (offset by one)
                x_seq = seq[:-1]
                y_seq = seq[1:]
                valid_lens.append(len(x_seq))
                pad_len = self.max_len - len(x_seq)
                if pad_len > 0:
                    x_padded = x_seq + [self.char_to_idx["<pad>"]] * pad_len
                    y_padded = y_seq + [self.char_to_idx["<pad>"]] * pad_len
                else:
                    x_padded = x_seq[:self.max_len]
                    y_padded = y_seq[:self.max_len]
            X_batch.append(x_padded)
            Y_batch.append(y_padded)
            X = torch.tensor(X_batch)
            Y = torch.tensor(Y_batch)
            valid_lens = torch.tensor(valid_lens)
            yield X, Y, valid_lens

class DataManager:
    @staticmethod
    def load_data(path, num_examples=None):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if num_examples:
            lines = lines[:num_examples]

        src_texts, tgt_texts = [], []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                src_texts.append(parts[0].strip())
                tgt_texts.append(parts[1].strip())
        return src_texts, tgt_texts

class Seq2SeqEngine:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device

    def train(self, data_loader, lr, num_epochs, char_to_idx):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(reduction="none") 
        #we'll mask pad tokens so they don't influence the loss

        self.model.train()
        loss_history = []
        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0
            num_batches = 0

            for batch in data_loader:
                enc_X, dec_X, dec_Y, src_valid_lens = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                # tgt_valid_lens is not being yielded by TransformerDataLoader, it only yields 4 elements
                Y_hat = self.model(enc_X, dec_X, src_valid_lens)
                l = loss_fn(Y_hat.reshape(-1, len(self.vocab)), dec_Y.reshape(-1))

                pad_idx = char_to_idx["<pad>"]
                mask = (dec_Y.reshape(-1)) != float(pad_idx)
                l = (l*mask).sum() / mask.sum()
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += l.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Time: {time.time()-start_time:.2f}")
        return loss_history

    def predict(self, src_sentence, char_to_idx, idx_to_char, max_len):
        self.model.eval()

        src_tokens = [char_to_idx["<bos>"]]
        for char in src_sentence:
            src_tokens.append(char_to_idx.get(char, char_to_idx["<unk>"]))
        src_tokens.append(char_to_idx["<eos>"])
        
        enc_X = torch.tensor(src_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        src_valid_lens = torch.tensor([len(src_tokens)], device=self.device)

        enc_output = self.model.encoder(enc_X, src_valid_lens)
        state = self.model.decoder.init_state(enc_output, src_valid_lens)
        dec_input = torch.tensor([char_to_idx["<bos>"]], dtype=torch.long, device=self.device).unsqueeze(0)
        output_seq = []
        
        for _ in range(max_len):
            Y, state = self.model.decoder(dec_input, state)
            
            prediction = Y.argmax(dim=2)[:, -1].item() #we take the last "predicted" word
            
            # Stop if <eos> is generated
            if prediction == char_to_idx["<eos>"]:
                break
                
            output_seq.append(prediction)
            # Prepare input for next step
            dec_input = torch.tensor([[prediction]], dtype=torch.long, device=self.device)
            
        decoded_sentence = "".join([idx_to_char[idx] for idx in output_seq])
        return decoded_sentence