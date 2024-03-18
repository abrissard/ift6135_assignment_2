import torch
import unittest
from lstm import LSTM

class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.batch_size = 32
        self.sequence_length = 5

        self.lstm = LSTM(self.input_size, self.hidden_size)

    def test_forward(self):
        inputs = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        hidden_states = (
            torch.randn(1, self.batch_size, self.hidden_size),
            torch.randn(1, self.batch_size, self.hidden_size)
        )

        outputs, new_hidden_states = self.lstm.forward(inputs, hidden_states)

        # Check the shape of the outputs
        self.assertEqual(outputs.shape, (self.batch_size, self.sequence_length, self.hidden_size))

        # Check the shape of the new hidden states
        self.assertEqual(new_hidden_states[0].shape, (1, self.batch_size, self.hidden_size))
        self.assertEqual(new_hidden_states[1].shape, (1, self.batch_size, self.hidden_size))

if __name__ == '__main__':
    unittest.main()