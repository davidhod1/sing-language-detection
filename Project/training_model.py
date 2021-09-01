from data_variables import *
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import itertools
import io


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)

    y_true = np.argmax(y_test, axis=1).tolist()

    cm = confusion_matrix(y_true, test_pred)

    figure = plot_confusion_matrix(cm, class_names=sign_actions)
    cm_image = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


labels = []
sequences = []
sign_labels = []

label_map = {label: num for num, label in enumerate(sign_actions)}

for action in sign_actions:
    for sequence in range(num_of_sequences):
        frame_values = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            frame_values.append(res)
        sequences.append(frame_values)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=4)

log_dir = os.path.join('Logs')
file_writer_cm = tf.summary.create_file_writer(log_dir)
tb_callback = TensorBoard(log_dir=log_dir)
callback = LambdaCallback(on_epoch_end=log_confusion_matrix)
cm_callback = callback

model = Sequential()
model.add(Dropout(0.2))
model.add(LSTM(24, return_sequences=True, input_shape=(60, 126)))
model.add(LSTM(24, return_sequences=False))
model.add(Dense(sign_actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, cm_callback],
          validation_data=(X_test, y_test))

model.save('sign_lang_detection.h5')

y_predict = model.predict(X_test)
y_true = np.argmax(y_test, axis=1).tolist()
y_predict = np.argmax(y_predict, axis=1).tolist()

accuracy = accuracy_score(y_true, y_predict)

print("ACCURACY: ", accuracy)
print("F1: ", f1_score(y_true, y_predict, average="micro"))
print("F1: ", f1_score(y_true, y_predict, average="macro"))