from data_generator import DataGenerator, ConalaDataGenerator
from transformer import Transformer, create_masks
from pdb import set_trace

import tensorflow as tf
import yaml
import time

config = yaml.safe_load(open("config.yml"))

MAX_LENGTH = 100
# data = DataGenerator("", "./data/atis")
data = ConalaDataGenerator("", "./data/conala-corpus")
model = Transformer(num_layers=6, d_model=512, num_heads=8, dff=2048,
                    input_vocab_size=data.tokenizer_nl.vocab_size + 2,
                    target_vocab_size=data.tokenizer_logic.vocab_size + 2,
                    pe_input=data.tokenizer_nl.vocab_size + 2,
                    pe_target=data.tokenizer_logic.vocab_size + 2, rate=0.1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)


save_model = bool(config["training"]["checkpoint"])
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if save_model and ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Restored latest checkpoint')


def evaluate(inp_sentence, tokenizer_nl, tokenizer_logic):
    start_token = [tokenizer_nl.vocab_size]
    end_token = [tokenizer_nl.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_nl.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_logic.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input,
                                               output,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_logic.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def translate(sentence, tokenizer_nl, tokenizer_logic):
    result, attention_weights = evaluate(sentence, tokenizer_nl, tokenizer_logic)
    predicted_sentence = tokenizer_logic.decode([i for i in result if i < tokenizer_logic.vocab_size])
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))


def validate(model, data, mode="valid"):
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    for inp, tar in data.batcher(mode=mode):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        predictions, _ = model(inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        valid_loss(loss)
        valid_accuracy(tar_real, predictions)
    return valid_accuracy.result().numpy(), valid_loss.result().numpy()


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp,
                                True,
                                enc_padding_mask,
                                combined_mask,
                                dec_padding_mask)

        loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)


for ep in range(config["training"]["num_epochs"]):
    s = time.perf_counter()

    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(data.batcher()):
        train_step(inp, tar)

        if batch % config["training"]["print_freq"] == 0:
            print("MB: {}, acc: {:.4f} loss: {:.4f}".format(
                batch + 1, train_accuracy.result().numpy(), train_loss.result().numpy()))

    valid_loss, valid_acc = validate(model, data)
    print("Validation {} acc: {:.4f} loss: {:.4f}".format(ep + 1, valid_loss, valid_acc))
    print("Time per epoch {:0.2f} seconds.".format(time.perf_counter() - s))

    if save_model and ep % config["training"]["save_freq"] == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(ep+1, ckpt_save_path))

test_loss, test_acc = validate(model, data, mode="test")
print("Test acc: {:.4f} loss: {:.4f}".format(test_loss, test_acc))

translate("How can I send a signal from a python program?", data.tokenizer_nl, data.tokenizer_logic)
translate("check if all elements in a list are identical", data.tokenizer_nl, data.tokenizer_logic)