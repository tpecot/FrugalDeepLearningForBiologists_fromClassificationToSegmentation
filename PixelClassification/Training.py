import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    from utils import train_unet

    return (train_unet,)


@app.cell
def _():
    import marimo as mo

    training_path = mo.ui.file_browser(selection_mode='directory', label="Input training path")
    model_path = mo.ui.file_browser(selection_mode='directory', label="Output model path")



    validation_proportion = mo.ui.number(value=0.2, step=0.01, label="Proportion of input data used for validation")
    bs = mo.ui.number(value=8, step=1, label="Batch size")
    nb_epochs = mo.ui.number(value=50, step=1, label="Number of epochs")
    augmentations = mo.ui.switch(value=True, label="Apply data augmentations")
    nb_epochs_without_improvement = mo.ui.number(value=5, step=1, label="Number of epochs without improvement before reducing learning rate")
    early_stopping = mo.ui.number(value=20, step=1, label="Number of epochs without improvement before early stopping")
    lr = mo.ui.number(value=0.001, step=0.000001, label="Learning rate")
    unet_depth = mo.ui.number(start=2, stop=6, value=4, step=1, label="UNet depth")
    imaging_field = mo.ui.number(start=64, stop=2048, value=256, step=64, label="Imaging field")
    nb_channels = mo.ui.number(start=1, stop=40, value=1, step=1, label="Number of channels")
    nb_classes = mo.ui.number(start=2, stop=10, value=3, step=1, label="Number of classes")
    classifier_name = mo.ui.text(label="Classifier name for QuPath")
    original_pixel_size = mo.ui.number(value=0.2, step=0.01, label="Original pixel size")
    downsample = mo.ui.number(value=1, step=1, label="Downsample")


    train_button = mo.ui.run_button(label="Train")
    return (
        augmentations,
        bs,
        classifier_name,
        downsample,
        early_stopping,
        imaging_field,
        lr,
        mo,
        model_path,
        nb_channels,
        nb_classes,
        nb_epochs,
        nb_epochs_without_improvement,
        original_pixel_size,
        train_button,
        training_path,
        unet_depth,
        validation_proportion,
    )


@app.cell
def _(mo, model_path, training_path):
    mo.vstack([training_path, model_path])
    return


@app.cell
def _(
    augmentations,
    bs,
    classifier_name,
    downsample,
    early_stopping,
    imaging_field,
    lr,
    mo,
    nb_channels,
    nb_classes,
    nb_epochs,
    nb_epochs_without_improvement,
    original_pixel_size,
    unet_depth,
    validation_proportion,
):
    mo.vstack([classifier_name, original_pixel_size, downsample, validation_proportion, unet_depth, imaging_field, nb_channels, nb_classes, nb_epochs, lr, bs, nb_epochs_without_improvement, early_stopping, bs, augmentations])
    return


@app.cell
def _(mo, nb_classes):
    classes = mo.ui.array(
        [
            mo.ui.text(
                label="Class " + str(i+1)
            )
            for i in range(nb_classes.value-1)
        ]
    )
    class_table = mo.ui.table(
        {
            "Class name": list(classes),
        }
    )
    class_table
    return (classes,)


@app.cell
def _(mo, nb_channels):
    channels = mo.ui.array(
        [
            mo.ui.number(
                value = i,
                label="Channnel " + str(i+1)
            )
            for i in range(nb_channels.value)
        ]
    )
    channel_table = mo.ui.table(
        {
            "Channnel number": list(channels),
        }
    )
    channel_table
    return (channels,)


@app.cell
def _(mo, train_button):
    mo.vstack([train_button])
    return


@app.cell
def _(
    augmentations,
    bs,
    channels,
    classes,
    classifier_name,
    downsample,
    early_stopping,
    imaging_field,
    lr,
    model_path,
    nb_channels,
    nb_classes,
    nb_epochs,
    nb_epochs_without_improvement,
    original_pixel_size,
    train_button,
    train_unet,
    training_path,
    unet_depth,
    validation_proportion,
):
    from pathlib import PurePath, PureWindowsPath
    if train_button.value:
        channel_def = []
        for i in range(len(channels.value)):
            channel_def.append({"channel": channels.value[i]})
        class_labels = {}
        random_colors = []
        random_colors.append([255,0,255])
        random_colors.append([0,255,255])
        random_colors.append([255,255,0])
        random_colors.append([128,0,255])
        random_colors.append([0,128,255])
        random_colors.append([128,255,0])
        random_colors.append([255,128,0])
        random_colors.append([0,255,128])
        random_colors.append([255,128,0])
        random_colors.append([255,255,255])
        for i in range(len(classes.value)):
            class_labels[str(i+1)] = {
                "name": classes.value[i],
                "color": random_colors[i]
            }

        train_unet(model_path=PureWindowsPath(model_path.path()).as_posix(), classifier_name=classifier_name.value, class_labels=class_labels, pixel_size=original_pixel_size.value*downsample.value, channel_def=channel_def,
               training_dir=PureWindowsPath(training_path.path()).as_posix(), validation_dir=None, test_size=validation_proportion.value, unet_depth=unet_depth.value, image_size=imaging_field.value, 
               n_channels=nb_channels.value, n_classes=nb_classes.value, epochs=nb_epochs.value, batch_size=bs.value, learning_rate=lr.value, 
               augmentations=augmentations.value, early_stopping=early_stopping.value, nb_epochs_without_improvement=nb_epochs_without_improvement.value)
    return


if __name__ == "__main__":
    app.run()
