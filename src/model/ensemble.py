conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.29-0.10.hdf5')
all_cnn_model.load_weights('weights/all_cnn.30-0.08.hdf5')
nin_cnn_model.load_weights('weights/nin_cnn.30-0.93.hdf5')

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]

def ensemble(models, model_input):

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model
ensemble_model = ensemble(models, model_input)
