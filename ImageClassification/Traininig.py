import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    from utils import train

    return (train,)


@app.cell
def _():
    import marimo as mo

    batch_size = mo.ui.number(value=8, step=1, label="Batch size")

    num_epochs = mo.ui.number(value=50, step=1, label="Number of epochs")

    nb_epochs_without_improvement = mo.ui.number(value=5, step=1, label="Number of epochs without improvement before reducing learning rate")

    data_dir = mo.ui.file_browser(selection_mode='directory', label="Input data path to train and val folders")

    output_path = mo.ui.file_browser(selection_mode='directory', label="Output data for trained model")

    lr = mo.ui.number(value=0.001, step=0.000001, label="Learning rate")
 
    frozen_network = mo.ui.switch(value=True, label="Freeze Network weights")

    augmentations = mo.ui.switch(value=True, label="Apply data augmentations")

    train_button = mo.ui.run_button(label="Train")
    return (
        augmentations,
        batch_size,
        data_dir,
        frozen_network,
        lr,
        mo,
        nb_epochs_without_improvement,
        num_epochs,
        output_path,
        train_button,
    )


@app.cell
def _(data_dir, mo, output_path):
    mo.vstack([data_dir, output_path])
    return


@app.cell
def _(
    augmentations,
    batch_size,
    frozen_network,
    lr,
    mo,
    nb_epochs_without_improvement,
    num_epochs,
):
    mo.vstack([num_epochs, lr, nb_epochs_without_improvement, batch_size, frozen_network, augmentations])
    return


@app.cell
def _(mo, train_button):
    mo.vstack([train_button])
    return


@app.cell
def _(
    augmentations,
    batch_size,
    data_dir,
    frozen_network,
    lr,
    nb_epochs_without_improvement,
    num_epochs,
    output_path,
    train,
    train_button,
):
    from pathlib import PurePath, PureWindowsPath
    if train_button.value:
        train(output_path=PureWindowsPath(output_path.path()).as_posix(), data_dir=PureWindowsPath(data_dir.path()).as_posix(), frozen_network=frozen_network.value, batch_size=batch_size.value, lr=lr.value, nb_epochs_without_improvement=nb_epochs_without_improvement.value, num_epochs=num_epochs.value, augmentations=augmentations.value)
    return


if __name__ == "__main__":
    app.run()
