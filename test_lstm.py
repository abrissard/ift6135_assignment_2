import torch
import unittest
from lstm import LSTM, Encoder

class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.sequence_length = 5
        self.hidden_size = 256

        self.encoder = Encoder()

    def test_encoder_forward(self):
        inputs = torch.randn(self.batch_size, self.sequence_length)
        hidden_states = (
            self.encoder.initial_states(self.batch_size)
        )

        outputs, new_hidden_states = self.encoder.forward(inputs, hidden_states)

        # Check the shape of the outputs
        self.assertEqual(outputs.shape, (self.batch_size, self.sequence_length, 2 * self.hidden_size))

        # Check the shape of the new hidden states
        self.assertEqual(new_hidden_states[0].shape, (2, self.batch_size, self.hidden_size))
        self.assertEqual(new_hidden_states[1].shape, (2, self.batch_size, self.hidden_size))


if __name__ == '__main__':
    unittest.main()