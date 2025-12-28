
import unittest
import torch
import os
import tempfile
import sys
from unittest.mock import MagicMock, patch

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.tmp_dir, 'checkpoint.pth')
        
        # Mock dependencies specifically for this test class
        self.modules_patcher = patch.dict(sys.modules, {
            'xgboost': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.linear_model': MagicMock(),
            'sklearn.metrics': MagicMock(),
            'sklearn.model_selection': MagicMock(),
            'cleanlab': MagicMock(),
            'cleanlab.classification': MagicMock(),
            'pulearn': MagicMock()
        })
        self.modules_patcher.start()
        
    def tearDown(self):
        self.modules_patcher.stop()
        import shutil
        shutil.rmtree(self.tmp_dir)
        
    def test_auto_detect_layers(self):
        # Import inside test to ensure mocks are active/reloaded if needed
        # We need to ensure evaluate is imported AFTER mocks are set
        if 'hierarchical.evaluation.evaluate' in sys.modules:
            del sys.modules['hierarchical.evaluation.evaluate']
        from hierarchical.evaluation.evaluate import load_model
        
        # Save a checkpoint with Day=2, Acc=4
        sd = self.create_dummy_state_dict(day_layers=2, acc_layers=4)
        torch.save(sd, self.ckpt_path)
        
        # Load with detection (num_layers=None)
        model = load_model(self.ckpt_path, device='cpu', hidden_dim=32, num_heads=4, num_layers=None)
        
        # Verify
        self.assertEqual(model.day_encoder.transformer.num_layers, 2)
        self.assertEqual(model.transformer.num_layers, 4)

    def test_force_layers(self):
        if 'hierarchical.evaluation.evaluate' in sys.modules:
            del sys.modules['hierarchical.evaluation.evaluate']
        from hierarchical.evaluation.evaluate import load_model

        # Save a checkpoint with Day=2, Acc=4
        sd = self.create_dummy_state_dict(day_layers=2, acc_layers=4)
        torch.save(sd, self.ckpt_path)
        
        # Force num_layers=3
        model = load_model(self.ckpt_path, device='cpu', hidden_dim=32, num_heads=4, num_layers=3)
        
        # Verify
        self.assertEqual(model.day_encoder.transformer.num_layers, 3)
        self.assertEqual(model.transformer.num_layers, 3)

    def create_dummy_state_dict(self, day_layers=2, acc_layers=4):
        # Create minimal state dict keys to simulate structure
        sd = {}
        # Transaction Encoder stuff (prefix)
        prefix = 'day_encoder.txn_encoder.'
        sd[f'{prefix}cat_group_emb.weight'] = torch.randn(10, 32)
        sd[f'{prefix}cat_sub_emb.weight'] = torch.randn(10, 32)
        # Default counter_party_dim is 64
        sd[f'{prefix}counter_party_emb.weight'] = torch.randn(10, 64)
        # balance_proj depends on use_balance. defaults?
        # Model default balance feature dim is 7. projection -> 1? No, projection -> 16 usually?
        # Actually TransactionEncoder default uses balance features size 7 -> 16 or similar?
        # Let's check error: "balance_proj.weight shape [32, 1] vs [16, 7]"
        # So model expects [16, 7] (7 input features, 16 output dim).
        sd[f'{prefix}balance_proj.weight'] = torch.randn(16, 7)
        sd[f'{prefix}balance_proj.bias'] = torch.randn(16)
        
        # Day Encoder Layers
        # Transformer default feedforward is 4*hidden_dim (32*4=128)
        for i in range(day_layers):
            sd[f'day_encoder.transformer.layers.{i}.linear1.weight'] = torch.randn(128, 32)
            sd[f'day_encoder.transformer.layers.{i}.linear1.bias'] = torch.randn(128)
            sd[f'day_encoder.transformer.layers.{i}.linear2.weight'] = torch.randn(32, 128)
            sd[f'day_encoder.transformer.layers.{i}.linear2.bias'] = torch.randn(32)
            
            # Attn weights
            # in_proj_weight is [3*embed_dim, embed_dim] -> [96, 32]
            sd[f'day_encoder.transformer.layers.{i}.self_attn.in_proj_weight'] = torch.randn(96, 32)
            sd[f'day_encoder.transformer.layers.{i}.self_attn.in_proj_bias'] = torch.randn(96)
            sd[f'day_encoder.transformer.layers.{i}.self_attn.out_proj.weight'] = torch.randn(32, 32)
            sd[f'day_encoder.transformer.layers.{i}.self_attn.out_proj.bias'] = torch.randn(32)
            
            # Norms
            sd[f'day_encoder.transformer.layers.{i}.norm1.weight'] = torch.randn(32)
            sd[f'day_encoder.transformer.layers.{i}.norm1.bias'] = torch.randn(32)
            sd[f'day_encoder.transformer.layers.{i}.norm2.weight'] = torch.randn(32)
            sd[f'day_encoder.transformer.layers.{i}.norm2.bias'] = torch.randn(32)

            
        # Account Encoder Layers
        for i in range(acc_layers):
            sd[f'transformer.layers.{i}.linear1.weight'] = torch.randn(128, 32)
            sd[f'transformer.layers.{i}.linear1.bias'] = torch.randn(128)
            sd[f'transformer.layers.{i}.linear2.weight'] = torch.randn(32, 128)
            sd[f'transformer.layers.{i}.linear2.bias'] = torch.randn(32)

            sd[f'transformer.layers.{i}.self_attn.in_proj_weight'] = torch.randn(96, 32)
            sd[f'transformer.layers.{i}.self_attn.in_proj_bias'] = torch.randn(96)
            sd[f'transformer.layers.{i}.self_attn.out_proj.weight'] = torch.randn(32, 32)
            sd[f'transformer.layers.{i}.self_attn.out_proj.bias'] = torch.randn(32)
            
            sd[f'transformer.layers.{i}.norm1.weight'] = torch.randn(32)
            sd[f'transformer.layers.{i}.norm1.bias'] = torch.randn(32)
            sd[f'transformer.layers.{i}.norm2.weight'] = torch.randn(32)
            sd[f'transformer.layers.{i}.norm2.bias'] = torch.randn(32)

        return sd

if __name__ == '__main__':
    unittest.main()
