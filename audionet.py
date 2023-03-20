import json
import os
import tensorflow as tf
from tensorflow import keras as keras

import matplotlib.pyplot as plt
import constants
import numpy as np
from IPython import display

class anetclass:
    def __init__(self) -> None:
        self.commands = self.get_commands()
        print("anetclass:   def __init__(self)")
        self.model = self.create_model()  
        
        pass
    
    # def retrain_model(self, command_dir, class_name):
        
    #     batch_size_train = 64
    #     epochs_train = 1
    #     early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1, min_delta=0.0)
    #     self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    #     # commands = [ f.name for f in os.scandir(constants.COMMANDS_AUDIO_PATH) if f.is_dir() ] 
        
    #     train_ds = tf.keras.utils.audio_dataset_from_directory(
    #     directory=command_dir,
    #     labels = None,
    #     batch_size=None,
    #     validation_split=None,
    #     seed=0,
    #     output_sequence_length=16000)      
        
        
    #     train_ds = train_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
    #     val_ds = val_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
                
    #     test_ds = val_ds.shard(num_shards=2, index=0)            
    #     for example_audio, example_labels in train_ds.take(1):  
    #         print(example_audio.shape)
    #         print(example_labels.shape)
            
    #     train_spectrogram_ds = self.make_spec_ds(train_ds) 
    #     train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)                                            
              
    #     train_ds = train_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
    #     for example_audio, example_labels in train_ds.take(1):  
    #         print(example_audio.shape)
    #         print(example_labels.shape)
    #     train_spectrogram_ds = self.make_spec_ds(train_ds)            
    #     history = self.model.fit(train_spectrogram_ds, validation_data = None, batch_size = batch_size_train, epochs = epochs_train, callbacks = [early_stop_callback])
           
    #     self.save_model(self.model)

    def train_model(self, model):
        print("##################################################")
        print("def train_model(self, model):")
        batch_size_train = 64
        epochs_train = 1
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1, min_delta=0.0)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        commands = [ f.name for f in os.scandir(constants.COMMANDS_AUDIO_PATH) if f.is_dir() ]
        # x_train =[]
        
        train_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=constants.COMMANDS_AUDIO_PATH,
        labels = "inferred",
        batch_size=1,
        validation_split=None,
        seed=0,
        output_sequence_length=16000)                       
        
        train_ds = train_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
        # val_ds = val_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
                
        # test_ds = val_ds.shard(num_shards=2, index=0)            
        # for example_audio, example_labels in train_ds.take(1):  
        #     print(example_audio.shape)
            # print(example_labels.shape)
            
        train_spectrogram_ds = self.make_spec_ds(train_ds) 
        train_spectrogram_ds = train_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)           
        
        # for example_spectrograms in train_spectrogram_ds.take(1):
        #     break        
        for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
            break

        input_shape = example_spectrograms.shape[1:]
        print('Input shape:', input_shape)
        num_labels = len(self.commands)
        
        print("input_shape= :" + str(input_shape))
        # write input_shape to config.json
        with open("config.json", "r+") as json_file:
            data = json.load(json_file)

            # data["input_shape"] = str(input_shape)
            data["input_shape"] = ([i for i in input_shape])
            data["num_labels"] = (num_labels)

            json_file.seek(0)  # rewind
            json.dump(data, json_file, indent = 2)
            json_file.truncate()
        
    
        history = model.fit(train_spectrogram_ds, batch_size = batch_size_train, epochs = epochs_train, callbacks = [early_stop_callback])   
        results = model.evaluate(train_spectrogram_ds, batch_size = batch_size_train)
        
        self.save_model(model)
        return model
    
              
    def train_model_save(self, model):
        print("##################################################")
        print("def train_model(self, model):")
        batch_size_train = 64
        epochs_train = 1
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1, min_delta=0.0)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        commands = [ f.name for f in os.scandir(constants.COMMANDS_AUDIO_PATH) if f.is_dir() ]
        # x_train =[]
        
        train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=constants.DATA_PATH,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')                       
        
        train_ds = train_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.util_squeeze, tf.data.AUTOTUNE)
                
        test_ds = val_ds.shard(num_shards=2, index=0)
        val_ds = val_ds.shard(num_shards=2, index=1)
        
        for example_audio, example_labels in train_ds.take(1):  
            print(example_audio.shape)
            print(example_labels.shape)
            
        train_spectrogram_ds = self.make_spec_ds(train_ds)
        val_spectrogram_ds = self.make_spec_ds(val_ds)
        test_spectrogram_ds = self.make_spec_ds(test_ds)      
        
        train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
        test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)      
        
        for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
            break

        input_shape = example_spectrograms.shape[1:]
        print('Input shape:', input_shape)
        num_labels = len(self.commands)
        
        print(input_shape)

        # write input_shape to config.json
        with open("config.json", "r+") as json_file:
            data = json.load(json_file)

            # data["input_shape"] = str(input_shape)
            data["input_shape"] = ([i for i in input_shape])
            data["num_labels"] = (num_labels)

            json_file.seek(0)  # rewind
            json.dump(data, json_file, indent = 2)
            json_file.truncate()
        
    
        history = model.fit(train_spectrogram_ds, validation_data = val_spectrogram_ds, batch_size = batch_size_train, epochs = epochs_train, callbacks = [early_stop_callback])   
        results = model.evaluate(test_spectrogram_ds, batch_size = batch_size_train)
        
        self.save_model(model)
        return model

    def util_squeeze(self, audio, labels):
        print("##################################################")        
        print("def util_squeeze(self, audio, labels):")
        # if (not audio or not labels):
        #     print("audio and labels")
        # This dataset only contains single channel audio, so use the tf.squeeze function to drop the extra axis:
        audio = tf.squeeze(audio, axis =-1)
        return audio, labels
        
    def classify(self, data):
        command =""
        return command
        
    def get_commands(self):
        commands = np.array([ f.name for f in os.scandir(constants.COMMANDS_AUDIO_PATH) if f.is_dir() ])
        # print(commands)
        return commands

    def util_get_spectogram(self, waveform):
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram   
    
    def make_spec_ds(self, ds):
        return ds.map(
            map_func=lambda audio,label: (self.util_get_spectogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)    
    
    def create_model(self):
        print("##################################################")
        print("def create_model(self):")
        model_dir = constants.MODEL_PATH
        # if (model_dir==9):
        if (os.path.exists(model_dir) and len(os.listdir(constants.MODEL_PATH))!= 0):
            print("class AudioNet: self.model = load_model(saved_model_path) ")
            # model = tf.keras.models.load_model(model_dir)               
            model = tf.keras.models.load_model(constants.MODEL_PATH+"mmodel.h5")
            print("model exists")
            
        else: 
            print("model doesnt exist")
            with open("config.json", "r") as config_file:
                data= json.load(config_file)           
                input_shape = list(data["input_shape"])
                output_shape = data["num_labels"]
    
                
            inputs = tf.keras.Input(shape = input_shape)
            # inputs = tf.keras.Input(shape = spectrogram.shape)
            x = tf.keras.layers.Rescaling(1.0 / 255)(inputs) 
            x = tf.keras.layers.Flatten()(x)
            
            n_neurons = 10
            activation = "relu"
            x = tf.keras.layers.Dense(n_neurons, activation = activation)(x)
            
            # n_outputs= n_commands
            outputs = tf.keras.layers.Dense(output_shape, activation)(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            # self.save_model(model)
            trained_model = self.train_model(model)
            return trained_model
                
        model.summary()   
             
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")     
        [print(i.shape, i.dtype) for i in model.inputs]
        [print(o.shape, o.dtype) for o in model.outputs]
        [print(l.name, l.input_shape, l.dtype) for l in model.layers]     
        return model
    
    def inference_plot(self, file_name):
        # x = data_dir/'no/01bb6a2a_nohash_0.wav'
        x= file_name
        x = tf.io.read_file(str(x))
        x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
        x = tf.squeeze(x, axis=-1)
        waveform = x
        x = self.util_get_spectogram(x)
        x = x[tf.newaxis,...]

        prediction = self.model(x)
        x_labels = self.commands 
        plt.bar(x_labels, tf.nn.softmax(prediction[0]))
        plt.title('Down')
        plt.show()

        display.display(display.Audio(waveform, rate=16000))     
        
    def classify(self, file_name):
        x= file_name
        x = tf.io.read_file(str(x))
        x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
        x = tf.squeeze(x, axis=-1)
        waveform = x        
        x = self.util_get_spectogram(x)
        x = x[tf.newaxis,...]         
        np_predictions = self.model.predict(x)
        
        best= 0
        y= 0
        for i in np_predictions:
            for j,_ in enumerate(i):
                if i[j] > best:
                    best=i[j]
                    print(best)
                    y=j
        return self.commands[y]
     
        
    def save_model(self, model):
        model.save(constants.MODEL_PATH+"mmodel.h5")