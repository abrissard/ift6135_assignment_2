import torch
import unittest
from transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.sequence_length = 10
        self.vocab_size = 100
        self.embed_dim = 256

        self.transformer = Transformer(self.vocab_size, self.embed_dim)

    def test_forward(self):
        inputs = torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
        mask = torch.ones((self.batch_size, self.sequence_length), dtype=torch.long)

        output = self.transformer.forward(inputs, mask)

        # Check the shape of the output
        self.assertEqual(output.shape, (self.batch_size, self.embed_dim))

        # Add more assertions if needed

if __name__ == '__main__':
    unittest.main()