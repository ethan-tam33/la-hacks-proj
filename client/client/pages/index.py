"""The home page of the app."""

from client import styles
from client.templates import template
import reflex as rx
from client.pages.analyze import AnalyzeState
import subprocess
import os
from PIL import Image

class UploadState(rx.State):
    "State of the uploaded images"

    # uploaded images
    img: list[str]
    count: int
    pred_score: int

    async def count_incr(self):
        self.count += 1

    async def handle_upload(self, files: list[rx.UploadFile]):
        """
        Handle the uploads of files.
        Args:
            files: list of uploaded files
        """
        for file in files:
            print(file)
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)

@template(route="/", title="Home")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """

    def get_first_file_path(directory):
        # List all files in the directory
        files = os.listdir(directory)
        if files:
            # Join directory path with the first file name to get the full path
            first_file_path = os.path.join(directory, files[0])
        return first_file_path

    def upload_and_update_count():
        return [UploadState.count_incr(), UploadState.handle_upload(rx.upload_files(upload_id="upload"))]

    def predict_and_analyze():
        first_img_path = get_first_file_path(rx.get_upload_dir())
        output = subprocess.run(["python", "../gemini/gemini_assessment.py", first_img_path, "1"],  stdout=subprocess.PIPE)
        output = output.stdout.decode('utf-8')
        img = Image.open(first_img_path)



        return [AnalyzeState.update_gemini(output)]

    return rx.container(
        rx.heading("Reefer", size="9"),
        rx.box(
            "Reefer uses machine learning to analyze a set of images of coral reefs - the most important organisms in the ocean.",
        ),
        rx.box(
            "Input your images of coral below. Please input at least 3 images.",
            text_align="middle",
        ),
    
        rx.center(rx.upload(
            rx.text(
                "Drag and drop images or click to select images here",
                text_align="middle"
            ),
            id="upload",
            border="1px dotted rgb(107,99,246)",
            padding="2em",
            width= "200px",
            #accept={'image/png':'.png'}
        )),
        rx.center(rx.box(
            f'Uploaded Images: {UploadState.count}'
        )),
        rx.hstack(rx.foreach(rx.selected_files("upload"), rx.text)),
        rx.center(rx.stack(
            rx.center(rx.button(
                "Upload",
                on_click=upload_and_update_count
            )),
            rx.center(rx.button(
                "Clear",
                on_click=rx.clear_selected_files("upload")
            )),
            rx.link(
                rx.center(rx.button(
                "Analyze",
                on_click=predict_and_analyze
                )),
                href='/analyze'
            ),
            flex_direction="column",
            align="center"
        )),
        rx.foreach(UploadState.img, lambda img: rx.image(src=rx.get_upload_url(img))),
        background="center/cover url('')"

    )
