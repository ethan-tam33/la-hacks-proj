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

    async def reset_count(self):
        self.count = 0
        self.img = []

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

    def get_first_file_path(self, directory):
        # List all files in the directory
        files = os.listdir(directory)
        if files:
            # Join directory path with the first file name to get the full path
            first_file_path = os.path.join(directory, files[0])
        return first_file_path

    # def clear_directory(directory):
    #     # Get list of all files in the directory
    #     files = os.listdir(directory)
        
    #     # Iterate over each file and delete it
    #     for file in files:
    #         file_path = os.path.join(directory, file)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)

    async def upload_and_update_count(self):
        #self.clear_directory("../rx.get_upload_dir()")
        print("uploads")
        return [UploadState.count_incr(), UploadState.handle_upload(rx.upload_files(upload_id="upload"))]

    async def predict_and_analyze(self):
        print("predict")
        first_img_path = self.get_first_file_path(rx.get_upload_dir())

        predict_output = subprocess.run(["python", "../recognition/predict.py", first_img_path], stdout=subprocess.PIPE)
        temp = predict_output.stdout.decode('utf-8').split('\n')[0].strip('()').split(',')
        is_healthy, confidence_score = [float(value.strip()) for value in temp]

        print(first_img_path)
        print(predict_output)

        is_healthy = int(is_healthy)
        confidence_score = float(confidence_score) * 100
        
        gemini_output = subprocess.run(["python", "../gemini/gemini_assessment.py", first_img_path, str(is_healthy), str(confidence_score)],  stdout=subprocess.PIPE)
        gemini_output = gemini_output.stdout.decode('utf-8')

        print(gemini_output)


        print(is_healthy)
        print(confidence_score)

        confidence_score = "Confidence score: " + str(confidence_score) + "%"
        if is_healthy:
            is_healthy = "Classification: Not Bleached"
        else:
            is_healthy = "Classification: Bleached"
        return [AnalyzeState.update_gemini(gemini_output), AnalyzeState.update_confidence_score(confidence_score), AnalyzeState.update_is_healthy(is_healthy), AnalyzeState.update_file_name(first_img_path)] 

    async def clear(self):
        print("clear")
        return [self.reset_count(), rx.clear_selected_files("upload")]

@template(route="/", title="Home")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """

    # def get_first_file_path(directory):
    #     # List all files in the directory
    #     files = os.listdir(directory)
    #     if files:
    #         # Join directory path with the first file name to get the full path
    #         first_file_path = os.path.join(directory, files[0])
    #     return first_file_path

    def upload_and_update_count():
        return [UploadState.count_incr(), UploadState.handle_upload(rx.upload_files(upload_id="upload"))]

    # def predict_and_analyze():
    #     first_img_path = get_first_file_path(rx.get_upload_dir())

    #     predict_output = subprocess.run(["python", "../recognition/predict.py", first_img_path], stdout=subprocess.PIPE)
    #     temp = predict_output.stdout.decode('utf-8').split('\n')[0].strip('()').split(',')
    #     is_healthy, confidence_score = [float(value.strip()) for value in temp]

    #     print(first_img_path)
    #     print(predict_output)
    #     print(6)

    #     is_healthy = int(is_healthy)
    #     confidence_score = float(confidence_score) * 100
        
    #     gemini_output = subprocess.run(["python", "../gemini/gemini_assessment.py", first_img_path, str(is_healthy), str(confidence_score)],  stdout=subprocess.PIPE)
    #     gemini_output = gemini_output.stdout.decode('utf-8')

    #     print(gemini_output)


    #     print(is_healthy)
    #     print(confidence_score)
    #     print("ya8y")

    #     confidence_score = "Confidence score: " + str(confidence_score) + "%"
    #     if is_healthy:
    #         is_healthy = "Classification: Not Bleached"
    #     else:
    #         is_healthy = "Classification: Bleached"
    #     #print(AnalyzeState.file_name)
    #     return [AnalyzeState.update_gemini(gemini_output), AnalyzeState.update_confidence_score(confidence_score), AnalyzeState.update_is_healthy(is_healthy), AnalyzeState.update_file_name(first_img_path)] 

    current_directory = os.getcwd()
    print(current_directory)

    # def clear():
    #     print("cleared")
    #     return [UploadState.reset_count(), rx.clear_selected_files("upload")]
    
    return rx.container(
        rx.image(
            src="Reefer.png",
            width="100px"
        ),
        rx.heading("Reefer", size="9"),
        rx.box(
            "Reefer uses machine learning to analyze a set of images of coral reefs - the most important organisms in the ocean.",
        ),
        rx.box(
            "Input your images of coral below.",
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
                on_click=UploadState.clear
                # remember to clear uploaded_files when button is clicked
            )),
            rx.link(
                rx.center(rx.button(
                "Analyze",
                on_click= UploadState.predict_and_analyze
                )),
                href='/analyze'
            ),
            flex_direction="column",
            align="center"
        )),
        rx.foreach(UploadState.img, lambda img: rx.image(src=rx.get_upload_url(img))),
        #background="center/cover url('background.png')"

    )
