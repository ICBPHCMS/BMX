import keras
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Reshape
from keras.models import Model
from reverse_gradient import GradientReversal


def get_trained_model(
        X_train,
        Y_train,
        bmass_train,
        n_vertex,
        sample_weights,
        n_dense=5,
        dense_units=25,
        dropout_rate=0.0,
        loss_weights=[1, 1],
        revert_grad=True):
    """
    n_dense = 5  # self normalizing network typically can be made deep without problems
    dense_units = 25  # roughly 1-2 times the input is "standard" thumb rule
    dropout_rate = 0.0
    """

    # self normalizing network a la Sepp Hochreiter, keep up the gradient (no
    # vanisching gradient, no batch-norm needed or RESNET)
    activation = 'selu'
    kernel_initializer = 'lecun_normal'
    optimizer = 'adam'
    # to us categorical enthropy
    num_classes = n_vertex + 1
    n_fetaures = X_train.shape[1]
    # This returns a tensor
    inputs = Input(shape=(n_fetaures,))

    # [nvertices*X_scaled_train.shape[1]], 1
    x = Reshape((n_fetaures, 1))(inputs)
    # assuming a flattened array with features of vertices behind each
    # other
    x = Conv1D(
        n_fetaures / n_vertex,
        n_fetaures / n_vertex,
        strides=n_fetaures / n_vertex,
        activation=activation,
        kernel_initializer=kernel_initializer)(x)
    # the x shape now is nvertices,X_scaled_train.shape[1] / n_vertex
    x = Conv1D(n_fetaures / n_vertex, 1,
               activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Conv1D(n_fetaures / n_vertex, 1,
               activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Conv1D(
        n_fetaures / n_vertex,
        1,
        activation=activation,
        kernel_initializer=kernel_initializer)(x)
    # last layer reduces dimension by 2
    x = Conv1D(
        n_fetaures / n_vertex / 2, 1,
        activation=activation,
        kernel_initializer=kernel_initializer)(x)
    x = Flatten()(x)

    # some common dense layer for classification and regression
    for i in range(n_dense):
        # a layer instance is callable on a tensor, and returns a
        # tensor
        x = Dense(
            dense_units,
            activation=activation,
            kernel_initializer=kernel_initializer)(x)
        x = Dropout(dropout_rate)(x)

    x_class = x  # layer for classification
    x_res = x  # layers for regression

    # revert gradient for mass regression
    if revert_grad:
        x_res = GradientReversal()(x_res)

    for i in range(5):
        x_class = Dense(dense_units, activation=activation,
                        kernel_initializer=kernel_initializer)(x_class)
        x_class = Dropout(dropout_rate)(x_class)

    class_predictions = Dense(num_classes, activation='softmax')(x_class)

    for i in range(5):
        x_res = Dense(
            dense_units,
            activation=activation,
            kernel_initializer=kernel_initializer)(x_res)
        x_res = Dropout(dropout_rate)(x_res)

    mass_regression = Dense(n_vertex, activation='linear')(x_res)
    api_model = Model(
        inputs=inputs,
        outputs=[
            class_predictions,
            mass_regression])

    api_model.compile(
        loss=[
            'categorical_crossentropy',
            'mean_squared_error'],
        optimizer=optimizer,
        metrics=['accuracy'],
        loss_weights=loss_weights)

    api_model.fit(X_train,
                  [Y_train, bmass_train[:, 0:n_vertex]],
                  batch_size=10000, epochs=15,
                  validation_split=0.05,
                  sample_weight=sample_weights)
    '''
    # larger batchsize similar to learning rate decay
    api_model.fit(X_train,
                  [Y_train, bmass_train[:, 0:n_vertex]],
                  batch_size=1000000,
                  epochs=10,
                  validation_split=0.05,
                  sample_weight=sample_weights)
    '''
    return api_model
