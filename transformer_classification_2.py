
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import os



def create_training_data(sequence_length):
    """
    """
    data_folder = './data/sentiment/aclImdb'
    train_folder = os.path.join(data_folder, 'train')
    test_folder  = os.path.join(data_folder,'test')

    seed = 77

    tf.random.set_seed(
        seed
    )

    batch_size = 32

    train_ds, val_ds    = keras.utils.text_dataset_from_directory(
        directory=train_folder
       ,label_mode = 'int'
       ,class_names = ['pos','neg']
       ,seed = 77
       ,validation_split = 0.1
       ,subset = 'both'
       ,batch_size=batch_size
    )

    text_only_train_ds = train_ds.map(lambda x,y: x)

    text_int_vectors    = keras.layers.TextVectorization(
        max_tokens=20000
       ,split='whitespace'
       ,output_mode = 'int'
      , output_sequence_length=sequence_length
    )
    text_int_vectors.adapt(text_only_train_ds)

    train_ds = train_ds.map(lambda x, y: (text_int_vectors(x),y))
    val_ds   = val_ds.map(lambda x, y: (text_int_vectors(x),y))

    return train_ds, val_ds, text_int_vectors


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads  = num_heads

        self.multi_head = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)

        self.dense_proj = keras.Sequential(
            [ layers.Dense(self.dense_dim, activation = "relu")
             ,layers.Dense(self.embed_dim)
            ]
        )

        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()

    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:,tf.newaxis,:]

        attention_output = self.multi_head(inputs, inputs,attention_mask=mask)
        projection_input = self.layer_norm1(inputs + attention_output)
        projection_output = self.dense_proj(projection_input)

        return self.layer_norm2(projection_input + projection_output)


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim":self.embed_dim
               ,"dense_dim":self.dense_dim
               ,"num_heads":self.num_heads
            }
        )
        return config


class PositionEmbedding(layers.Layer):
    """

    """
    def __init__(self,sequence_length, vocab_size, embd_dim,**kwargs):
        super().__init__(**kwargs)
        self.token_embdngs = layers.Embedding(vocab_size,embd_dim)
        self.position_embdngs = layers.Embedding(sequence_length,embd_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim

    def call(self,inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0,limit=length,delta=1)
        embd_tokens = self.token_embdngs(inputs)
        embd_positions = self.position_embdngs(positions)
        return embd_tokens + embd_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length
           ,"vocab_size": self.vocab_size
           ,"embd_dim": self.embd_dim
        })
        return config


def build_model(sequence_length):
    vocab_size = 20000
    embed_dim = 256
    dense_dim = 32
    num_heads = 3

    inputs = keras.Input(shape=(None,),dtype=tf.int64)
    x = PositionEmbedding(vocab_size, sequence_length, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy",metrics=["accuracy"])
    return model

sequence_length=200

train_ds, val_ds, text_int_vec = create_training_data(sequence_length)
model = build_model(sequence_length)
callbacks = [keras.callbacks.ModelCheckpoint('./model/trans_classifc/transformer_encoder-1.keras', save_best_only=True)]
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
