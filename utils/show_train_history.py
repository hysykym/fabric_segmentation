import matplotlib.pyplot as plt


def show_train_history(train_history, train='loss', validation='val_loss'):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.xticks(range(0, train_history.params['epochs']))
    plt.title('Train History')
    plt.ylabel('LOSS')
    plt.xlabel('EPOCHS')
    plt.legend(['train', 'validation'])
    plt.show()


if __name__ == "__main__":
    pass
