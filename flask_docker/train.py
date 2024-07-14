import numpy
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

poly1d_clf = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

import joblib
joblib.dump(poly1d_clf, "poly1d_clf.joblib")
