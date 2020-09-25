from keras.models import load_model

# model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_53267_1_043_0.0601_0.0607_0.9754_0.9761.h5') # U-net++ lstm
model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_28983_0_031_0.0656_0.0619_0.9748_0.9764.h5') # U-net lstm
model.summary()