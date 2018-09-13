from keras.layers import Dense, Input
from keras.models import Model, model_from_json
import pickle


ver_path = ""


class AutoEncoder(object):
    def __init__(self, input_dim=None, encoding_dim=8):
        self._autoencoder = None
        self._encoder = None
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def set_encoding_dim(self, dim):
        self.encoding_dim = dim

    def set_input_dim(self, dim):
        self.input_dim = dim

    def _construct_shallow(self):
        input_ = Input(shape=(self.input_dim, ))
        encoded = Dense(self.encoding_dim, activation='relu')(input_)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)

        self._autoencoder = Model(input_, decoded)
        self._encoder = Model(input_, encoded)
        self._autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    def _construct_deep(self):
        input_ = Input(shape=(self.input_dim, ))
        encoded = Dense(64, activation='relu')(input_)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)

        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        self._autoencoder = Model(input_, decoded)
        self._encoder = Model(input_, encoded)
        self._autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    def construct_ae_components(self, model_type='deep'):
        if model_type == 'deep':
            self._construct_deep()
        elif model_type == 'shallow':
            self._construct_shallow()

    def fit(self, X_train, X_test, epochs=20, batch_size=256):
        self._autoencoder.fit(X_train, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    def encode(self, X):
        return self._encoder.predict(X)

    def save_model(self, full_path=ver_path):
        """ Save AutoEncoder data to disk """
        ae_vars = {'input_dim': self.input_dim, 'encoding_dim': self.encoding_dim}
        with open(full_path + "ae_vars.pickle", "wb") as f:
            pickle.dump(ae_vars, f)

        # Serialize models to JSON
        encoder_json = self._encoder.to_json()
        autoencoder_json = self._autoencoder.to_json()

        with open(full_path + "auto_encoder.json", "w") as json_file:
            json_file.write(autoencoder_json)
        with open(full_path + "encoder.json", "w") as json_file:
            json_file.write(encoder_json)

        # Serialize weights to HDF5
        self._autoencoder.save_weights(full_path + "auto_encoder.h5")
        self._encoder.save_weights(full_path + "encoder.h5")
        print("Saved model to disk")

    @staticmethod
    def load_model(full_path=ver_path):
        """ Instantiate AutoEncoder with data saved to disc. Returns AutoEncoder"""

        # Load jsons and pickles
        with open(full_path + "auto_encoder.json", 'r') as json_file:
            auto_encoder_json = json_file.read()
        with open(full_path + "encoder.json", 'r') as json_file:
            encoder_json = json_file.read()
        with open(full_path + "ae_vars.pickle", 'rb') as f:
            ae_vars = pickle.load(f)

        ae = AutoEncoder(input_dim=ae_vars['input_dim'], encoding_dim=ae_vars['encoding_dim'])
        ae._autoencoder = model_from_json(auto_encoder_json)
        ae._encoder = model_from_json(encoder_json)

        # Load weights into models
        ae._autoencoder.load_weights(full_path + "auto_encoder.h5")
        ae._encoder.load_weights(full_path + "encoder.h5")
        print("Loaded model from disk")
        return ae


if __name__ == '__main__':
    pass