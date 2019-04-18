from keras import models
from keras import layers
from keras.optimizers import SGD

model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(28, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd)

model.fit(X_train, y_train, epochs=5, batch_size=2000)

preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
