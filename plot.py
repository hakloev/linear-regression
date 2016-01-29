from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_history(history, save=False):
    plt.plot(history)
    plt.axis([-5, len(history) + 10, min(history) - 0.001, max(history) + 0.001])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    if save:
        plt.savefig('loss_history.png')
    else:
        plt.show()


def plot_regression(x, y, lr, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], y)
    print('Scattered data')

    x_surf = np.arange(0, 1, 0.001 * 25)
    y_surf = np.arange(0, 1, 0.001 * 25)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    exog = pd.core.frame.DataFrame({'X': x_surf.ravel(), 'Y': y_surf.ravel() })
    predictions = []
    #  print('# Starting creating predictions for hyperplane')
    for index, row in exog.iterrows():
        predictions.append(lr.predict([row.X, row.Y]))
    #  print('# Done creating predictions for hyperplane')
    predictions = np.array(predictions).reshape(x_surf.shape)
    #  print('# Plotting hyperplane')
    ax.plot_surface(x_surf, y_surf,
                    predictions,
                    rstride=1000,
                    cstride=1000,
                    color='None',
                    alpha=0.4)
    #  print('# Done plotting hyperplane')

    if save:
        plt.savefig('regression.png')
    else:
        plt.show()
