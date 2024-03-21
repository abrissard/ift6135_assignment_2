import torch
import unittest
from transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.sequence_length = 10
        self.num_heads = 4
        self.head_size = 64

        self.transformer = Transformer(num_heads=self.num_heads, head_size=self.head_size)

    def test_forward(self):
        hidden_states = torch.randn(self.batch_size, self.sequence_length, self.num_heads * self.head_size)
        mask = torch.ones(self.batch_size, self.sequence_length).long()

        outputs = self.transformer.forward(hidden_states, mask)

        # Check the shape of the outputs
        self.assertEqual(outputs.shape, (self.batch_size, self.sequence_length, self.num_heads * self.head_size))

        # Add more assertions to validate the correctness of the outputs

if __name__ == '__main__':
    unittest.main()