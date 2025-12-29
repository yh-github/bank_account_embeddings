import unittest
import torch
import torch.nn as nn
from hierarchical.models.day import DayEncoder
from hierarchical.models.transaction import TransactionEncoder

class TestDayEncoderNaN(unittest.TestCase):
    def test_nan_on_empty_day(self):
        """Verify DayEncoder does not produce NaNs for days with 0 transactions (fully masked)."""
        
        # Mocks
        input_dim = 16
        hidden_dim = 16
        
        # Dummy TransactionEncoder
        class MockTxnEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_dim = input_dim
            def forward(self, *args, **kwargs):
                # Return dummy embeddings [Batch, Days, Txns, Dim] -> [Batch*Days*Txns, Dim] ??
                # DayEncoder expects [cat_group, etc]
                # And calls txn_encoder.
                pass
        
        # Actually, let's use real encoders or mocked output
        # DayEncoder calls txn_encoder.
        # We can bypass txn_encoder for this test if we can mock it?
        # But DayEncoder initializes it.
        
        # Easier: Create a DayEncoder and mock the txn_encoder.forward
        txn_encoder = TransactionEncoder(
            num_categories_group=10, num_categories_sub=10, num_counter_parties=10, embedding_dim=input_dim
        )
        day_encoder = DayEncoder(txn_encoder, hidden_dim=hidden_dim, num_layers=1, num_heads=4)
        
        # Input Data
        # Batch=1, Days=2. Day 1 has data. Day 2 is EMPTY (0 transactions).
        # We simulate this by masking Day 2.
        
        # Dimensions
        B, D, T = 1, 2, 5
        
        # Create random inputs
        cat_group = torch.randint(0, 10, (B, D, T))
        cat_sub = torch.randint(0, 10, (B, D, T))
        amounts = torch.randn(B, D, T).unsqueeze(-1) # [B, D, T, 1]
        
        # Valid date indices: [0..6, 1..31, 1..12]
        dates_dow = torch.randint(0, 7, (B, D, T, 1)).float()
        dates_dom = torch.randint(1, 28, (B, D, T, 1)).float()
        dates_mon = torch.randint(1, 12, (B, D, T, 1)).float()
        dates = torch.cat([dates_dow, dates_dom, dates_mon], dim=-1) # [B, D, T, 3]

        cat_cp = torch.randint(0, 10, (B, D, T))
        
        # MASK:
        # Day 1: All Valid (True)
        # Day 2: All Invalid (False) -> This is the "Empty Day" case
        mask = torch.ones(B, D, T, dtype=torch.bool)
        mask[:, 1, :] = False # Sets Day 2 transactions to be IGNORED (False = Padding??)
        
        # Wait, DayEncoder expects input_mask where True = Keep, False = Pad?
        # Let's check day.py logic.
        # key_padding_mask = ~input_mask
        
        # So setting mask[:, 1, :] = False makes Day 2 fully padded.
        
        has_data = torch.tensor([[True, False]]) # Day 2 mark as invalid
        
        # MIMIC AccountEncoder: Flatten first 2 dims
        def flatten(x):
            return x.reshape(B*D, *x.shape[2:]) if x is not None else None
            
        stream_data = {
            'cat_group': flatten(cat_group),
            'cat_sub': flatten(cat_sub),
            'cat_cp': flatten(cat_cp),
            'amounts': flatten(amounts),
            'dates': flatten(dates),
            'mask': flatten(mask),
            'has_data': has_data.flatten()
        }
        
        # Forward
        # We only check _encode_stream as it contains the logic
        output = day_encoder._encode_stream(stream_data)
        
        # Check for NaNs
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs!")
        
        # Check that Day 2 (Index 1) is exactly zero (masked out)
        # Output is [B*D, Dim]
        self.assertTrue((output[1, :] == 0).all(), "Masked day should be zero")

if __name__ == '__main__':
    unittest.main()
