from main import load_data, preprocess_data, train_model, evaluate_model


def test_load_data():
    df = load_data()
    assert df.shape == (150, 5)


def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)


def test_train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    assert model is not None


def test_evaluate_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.7
