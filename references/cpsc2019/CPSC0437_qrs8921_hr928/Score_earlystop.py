from score_model import score_model_new, score_model
from keras import optimizers
from keras.callbacks import LearningRateScheduler

def Score_earlystop_new(model, 
				X_train, Y_train,
				X_test, Y_test,
				x_validation, QRS_validation,
				batch_size,
				epochs,
				class_weights,
				patience,
				best_model_file,
				fs=500,
				padding_length=2048):

	Score = 0
	current_patience = 0

	for e in range(epochs):

		model.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(X_test, Y_test),
				  shuffle=True,
				  class_weight=class_weights,
				  verbose=1)

		total_score = \
			score_model_new(model, x_validation, QRS_validation, 0.25, verbose=False,
				seg_length=fs*4, step_length=fs, padding_length=padding_length, fs=fs)

		print('Total score:', total_score)

		if total_score > Score:
			Score = total_score
			model.save_weights(best_model_file)
			current_patience = 0
		else:
			current_patience += 1

		if current_patience > patience:
			# stop training
			break

	return Score


def Score_earlystop_varyLR(model, 
				X_train, Y_train,
				X_test, Y_test,
				x_validation, QRS_validation,
				batch_size,
				epochs,
				class_weights,
				patience,
				best_model_file,
				fs=500,
				padding_length=2048,
				lr_patience=3):


	Score = 0
	current_patience = 0
	lr = [0.001]

	# learning rate schedule
	def step_decay(epoch):
		if current_patience>1 and current_patience%lr_patience == 0:
			lr[0] = lr[0] / 2
			model.load_weights(best_model_file)

		print('lr: ', lr[0])
		return lr[0]

	step_decay(0)

	# learning schedule callback
	lrate = LearningRateScheduler(step_decay)

	for e in range(epochs):

		model.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(X_test, Y_test),
				  shuffle=True,
				  class_weight=class_weights,
				  verbose=1,
				  callbacks=[lrate])

		total_score = \
			score_model_new(model, x_validation, QRS_validation, 0.25, verbose=False,
				seg_length=fs*4, step_length=fs, padding_length=padding_length, fs=fs)

		print('Total score:', total_score)

		if total_score > Score:
			Score = total_score
			model.save_weights(best_model_file)
			current_patience = 0
		else:
			current_patience += 1

		print('current_patience: ', current_patience)

		if current_patience > patience:
			# stop training
			break

	return Score



def Score_earlystop(model, 
				X_train, Y_train,
				X_test, Y_test,
				x_validation, QRS_validation,
				batch_size,
				epochs,
				class_weights,
				patience,
				best_model_file):

	Score = 0
	current_patience = 0

	for e in range(epochs):

		model.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(X_test, Y_test),
				  shuffle=True,
				  class_weight=class_weights,
				  verbose=1)

		total_score = \
			score_model(model, x_validation, QRS_validation, 0.25, verbose=False)

		print('Total score:', total_score)

		if total_score > Score:
			Score = total_score
			model.save_weights(best_model_file)
			current_patience = 0
		else:
			current_patience += 1

		if current_patience > patience:
			# stop training
			break

	return Score