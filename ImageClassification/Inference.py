import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    from utils import inference_pipeline

    return (inference_pipeline,)


@app.cell
def _():
    import marimo as mo

    model_path = mo.ui.file_browser(selection_mode='file', label="Input model")

    image_path = mo.ui.file_browser(selection_mode='directory', label="Path to images for classification")

    train_path = mo.ui.file_browser(selection_mode='directory', label="Train input data path")

    inference_button = mo.ui.run_button(label="Inference")
    return image_path, inference_button, mo, model_path, train_path


@app.cell
def _(image_path, mo, model_path, train_path):
    mo.vstack([train_path, model_path, image_path])
    return


@app.cell
def _(
    image_path,
    inference_button,
    inference_pipeline,
    model_path,
    train_path,
):
    from pathlib import PurePath, PureWindowsPath
    if inference_button.value:
        inference_pipeline(PureWindowsPath(model_path.path()).as_posix(), PureWindowsPath(image_path.path()).as_posix(), PureWindowsPath(train_path.path()).as_posix())
    return


@app.cell
def _(inference_button, mo):
    mo.vstack([inference_button])
    return


if __name__ == "__main__":
    app.run()
