"""The home page of the app."""

from client import styles
from client.templates import template
import reflex as rx

class UploadState(rx.State):
    "State of the uploaded images"

    # uploaded images
    img: list[str]
    count: int

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
    def upload_and_update_count():
        return [UploadState.count_incr(), UploadState.handle_upload(rx.upload_files(upload_id="upload"))]

    return rx.container(
        rx.heading("Reefer", size="9"),
        rx.box(
            "Reefer uses machine learning to analyze a set of images of coral reefs - the most important organisms in the ocean.",
            text_align="middle",
        ),
        rx.box(
            " "
        ),
        rx.box(
            "Input your images of coral below. Please input at least 3 images.",
            text_align="middle",
        ),
        rx.upload(
            rx.text(
                "Drag and drop images here or click to select images"
            ),
            id="upload",
            border="1px dotted rgb(107,99,246)",
            padding="5em",
            text_align="middle"
            #accept={'image/png':'.png'}
        ),
        rx.box(
            f'Uploaded Images: {UploadState.count}'
        ),
        rx.hstack(rx.foreach(rx.selected_files("upload"), rx.text)),
        rx.vstack(
            rx.button(
                "Upload",
                on_click=upload_and_update_count
            ),
            rx.button(
                "Clear",
                on_click=rx.clear_selected_files("upload"),
            ),
        ),
        rx.foreach(UploadState.img, lambda img: rx.image(src=rx.get_upload_url(img))),

    )
    # with open("README.md", encoding="utf-8") as readme:
    #     content = readme.read()
    # return rx.markdown(content, component_map=styles.markdown_style)
