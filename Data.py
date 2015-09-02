import numpy as np


def partition(X,y):
    np.random.seed(2)
###################### Partition the Data appropriately ##########################
    permutation = np.random.choice(range(X.shape[ 0 ]),
    X.shape[ 0 ], replace = False)
    size_train = np.round(X.shape[ 0 ] * 0.6)
    size_dev = np.round(X.shape[ 0 ] * 0.8)

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train :size_dev]
    index_dev = permutation[ size_dev : ]

    X_train = X[ index_train, : ]
    y_train = y[ index_train ]

    X_dev = X[ index_dev, : ]
    y_dev = y[ index_dev ]

    X_test = X[ index_test, : ]
    y_test = y[ index_test ]

    print X_train.shape
    print X_dev.shape
    print X_test.shape
    return {'X_train':X_train,'y_train':y_train,
            'X_dev': X_dev,'y_dev':y_dev,
            'X_test':X_test,'y_test':y_test}
