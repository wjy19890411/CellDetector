import keras_model


def main():
    model = keras_model.build_model()
    keras_model.fit(model)
    pred = keras_model.predict(model)
    print(pred.shape)


if __name__ == '__main__':
    main()
