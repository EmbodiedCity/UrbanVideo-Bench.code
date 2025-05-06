import cv2
import base64
import time
from openai import OpenAI
import os
import pandas as pd
import warnings
import math
import copy

warnings.filterwarnings("ignore")


# Main execution block
if __name__ == '__main__':
    # Need to input
    # Define the model name and initialize the OpenAI client with API credentials.
    model = "xxx"
    client = OpenAI(
        api_key='xxxxxxxxxxxxxx',  # Replace with your OpenAI API key.
        base_url='xxxxxxxxxxxxxx'  # Replace with your OpenAI API base URL.
    )

    # Dataset path
    folder_path = 'dataset/videos'  # Define the folder path where video files are stored.
    QA_df = pd.read_parquet('dataset/MCQ.parquet')  # Read the dataset containing questions and metadata from a Parquet file.

    # Define the folder path for saving results and create it if it doesn't exist.
    folder_path_result = 'result'
    if not os.path.exists(folder_path_result):
        os.makedirs(folder_path_result)

    # Define the path for the result CSV file.
    res_path = os.path.join(folder_path_result, '%s_output.csv' % model)

    # Check if the result file already exists.
    if os.path.exists(res_path):
        # If the file exists, load it and find the last valid index in the 'Output' column.
        res = pd.read_csv(res_path, index_col=0)
        last_valid_index = int(res['Output'].last_valid_index())
        last_valid_index += 1  # Start processing from the next index.
    else:
        # If the file doesn't exist, create a new DataFrame based on the QA dataset.
        res = copy.deepcopy(QA_df)
        res['Output'] = None  # Add an 'Output' column initialized to None.
        last_valid_index = 0  # Start processing from the first index.

    # Iterate through each question starting from the last valid index.
    for qa_idx in range(last_valid_index, res.shape[0]):
        print('Processing index: %d' % qa_idx)

        # Get the video ID for the current question.
        select_vid_name = res['video_id'].iloc[qa_idx]

        # Open the video file using OpenCV.
        video = cv2.VideoCapture(os.path.join(folder_path, str(select_vid_name)))
        video_fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video.

        # Initialize a list to store base64-encoded frames.
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()  # Read a frame from the video.
            if not success:
                break  # Stop reading if there are no more frames.

            # Encode the frame as a JPEG image and convert it to base64 format.
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        # Release the video file and print the number of frames read.
        video.release()
        print(len(base64Frames), "frames read.")

        # Create a prompt for the GPT model to answer questions based on the video.
        prompt = "This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions: \n"
        prompt += "The template for the answer is: \n\
                        Option: []; Reason: []\n\
                        where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option."

        # Add the question from the dataset to the prompt.
        qa = res['question'].iloc[qa_idx]
        prompt += '\n' + qa

        try:
            # Select a subset of frames to reduce the number of frames sent to the model.
            div_num = math.ceil(len(base64Frames) / 32)  # Divide frames into chunks.
            base64Frames_selected = base64Frames[0::div_num]  # Select every nth frame.

            # Prepare the video content in base64 format for the GPT model.
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            for buffer in base64Frames_selected:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{buffer}"
                    }})

            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Send the prompt to the GPT model and get the response.
            result = client.chat.completions.create(
                model=model,
                messages=PROMPT_MESSAGES
            )
            print(result.choices[0].message.content)  # Print the model's response.
            res_str = result.choices[0].message.content  # Extract the response string.

            # Save the response in the 'Output' column of the result DataFrame.
            res['Output'].iloc[qa_idx] = res_str

        except Exception as e:
            # Handle errors and wait for 60 seconds before retrying.
            print(f"An error occurred: {e}")
            time.sleep(60)

        # Save the updated result DataFrame to the CSV file.
        res.to_csv(res_path)