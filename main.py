from data_generator import DataGenerator
from transformer import Transformer, create_masks
from pdb import set_trace

import tensorflow as tf

data = DataGenerator("", "./data/atis")
model = Transformer(num_layers=4, d_model=512, num_heads=2, dff=2048,
					input_vocab_size=data.tokenizer_nl.vocab_size + 2,
					target_vocab_size=data.tokenizer_logic.vocab_size + 2,
					pe_input=data.tokenizer_nl.vocab_size + 2,
					pe_target=data.tokenizer_logic.vocab_size + 2, rate=0.1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
	name='train_accuracy')

def validate(model, data):
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    for inp, tar in data.batcher(mode="valid"):
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


optimizer = tf.keras.optimizers.Adam(learning_rate=.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

for ep in range(100):
	for (batch, (inp, tar)) in enumerate(data.batcher()):
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
			print("MB: {:.4f}, acc: {:.4f} loss: {:.4f}".format(
				batch + 1, train_accuracy.result().numpy(), train_loss.result().numpy()))

	valid_loss, valid_acc = validate(model, data)
	print("Validation {} acc: {:.4f} loss: {:.4f}".format(ep + 1, valid_loss, valid_acc))

