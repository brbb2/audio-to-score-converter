import unittest
from neural_network_trainer import *
from keras.utils import to_categorical


class TestNeuralNetworkTrainer(unittest.TestCase):

    def setUp(self):
        self.x, self.y = get_data(encoding='midi_pitch')
        self.x_train, self.y_train, self.x_val, self.y_val = split_data(self.x, self.y)

        self.x_train_reshaped = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)

        self.y_train_adjusted = self.y_train - 20
        mask_train = np.where(self.y_train_adjusted == -21)
        self.y_train_adjusted[mask_train] = 0
        self.y_train_one_hot = to_categorical(self.y_train_adjusted, num_classes=89, dtype='float32')

        self.x_val_reshaped = self.x_val.reshape(self.x_val.shape[0], self.x_val.shape[1], 1)

        self.y_val_adjusted = self.y_val - 20
        mask_val = np.where(self.y_val_adjusted == -21)
        self.y_val_adjusted[mask_val] = 0
        self.y_val_one_hot = to_categorical(self.y_val_adjusted, num_classes=89, dtype='float32')

    def test_reduce_data(self):
        proportion = 0.25
        x_reduced, y_reduced = reduce_data(self.x, self.y, proportion=proportion)
        print(f'reduced data-set size: {len(x_reduced):>6}   expecting: {len(self.x) * proportion:10.3f}')
        self.assertEqual(len(x_reduced), len(y_reduced))
        self.assertAlmostEqual(len(x_reduced), len(self.x) * proportion, delta=1)

    def test_data_split(self):
        print()
        print(f'         x_train.shape: {str(self.x_train.shape):<18} '
              f'x_train_reshaped.shape: {self.x_train_reshaped.shape}\n'
              f'         y_train.shape: {str(self.y_train.shape):<18} '
              f' y_train_one_hot.shape: {self.y_train_one_hot.shape}\n\n'
              f'           x_val.shape: {str(self.x_val.shape):<18} '
              f'  x_val_reshaped.shape: {self.x_val_reshaped.shape}\n'
              f'           y_val.shape: {str(self.y_val.shape):<18} '
              f'   y_val_one_hot.shape: {self.y_val_one_hot.shape}\n')
        print(f'\nx_train: {self.x_train.shape}')
        print(self.x_train)
        print(f'\nx_train_reshaped: {self.x_train_reshaped.shape}')
        print(self.x_train_reshaped)
        print(f'\ny_train: {self.y_train.shape}')
        print(self.y_train)
        print(f'\ny_train_adjusted: {self.y_train_adjusted.shape}')
        print(self.y_train_adjusted)
        print(f'\ny_train_one_hot: {self.y_train_one_hot.shape}')
        print(self.y_train_one_hot)
        print(f'\nx_val: {self.x_val.shape}')
        print(self.x_val)
        print(f'\nx_val_reshaped: {self.x_val_reshaped.shape}')
        print(self.x_val_reshaped)
        print(f'\ny_val: {self.y_val.shape}')
        print(self.y_val)
        print(f'\ny_val_adjusted: {self.y_val_adjusted.shape}')
        print(self.y_val_adjusted)
        print(f'\ny_val_one_hot: {self.y_val_one_hot.shape}')
        print(self.y_val_one_hot)


if __name__ == '__main__':
    unittest.main()
