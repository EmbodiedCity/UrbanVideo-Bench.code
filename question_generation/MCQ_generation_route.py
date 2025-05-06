import time
import os
import re
import pandas as pd
import pickle
import json

# This script is used for generating multiple-choice questions (MCQs) of Route Oriented Navigation Tasks for given videos using the Gemini model. For more detail of the Route Oriented Navigation Tasks, please refer to our paper.

question_categories = [
    "Progress Evaluation",
    "Landmark Position",
    "Action Generation"
]

def make_prompt(question_category, movement_instructions):
    
    MAIN_PROMPT = f"""
    MAIN INSTRUCTIONS:
        You are a marvoulous embodied UAV AI flying in urban areas. Your goal is to recall and comprehend the in-coming video input taken by your camera, and try to think like humans do to control your movement.
        When you think about the movements and navigation, you should raise textual questions around the video input, so as to help your thinking, and also to help the engineers that developed you to understand what you think. Your goal is to raise multi choice questions about the details or logic in the video, and then generate proper choices and the correct answer. You must strictly follow the later-on given requirements.
        Remember, you are currently flying in urban environments, following the movement instructions given to you.
        Now, you are required to raise a {question_category} question based what you've seen. 
        You are following these instructions: {movement_instructions}.
        You should generate a question based on the video input together with the objects, places and actions shown in it.
        For this kind of question, the special requirements are as follows:
    """

    NOTICE = f"""
    Always use second person perspective when generating questions. For example, use “You are moving following a series of movement instructions.” instead of “The UAV is moving following a series of movement instructions.”.
    Always use first person perspective when generating the choices. For example, use “I am flying over the buildings.” instead of “The UAV is flying over the buildings.”.
    Always use present tense when generating the questions and the choices. For example, use “What are your current move?” instead of “What was the UAV's current move?”.
    NEVER add extra explanatory text to the output. The output should only contain the Question, the Choices and the Answer, exactly in the form given above.
	When you fail to generate the question, simply output “Question Generation Failure”.
    When you try to describe directions or positional relationship, you should use left, right, front, back, etc, instead of north, south, east, west.
    ALWAYS remember to generate options that are different from the examples above, and also different from each other.
    """

    match question_category:
        case "Progress Evaluation":
            category_prompt =  f"""
                To generate a Progress Evaluation question, let's begin thinking step by step.
                To help you understand, this is a question that requires you to recall your movements shown in the video, then align these movements with the given movement instructions, and finally figure out which movement was last taken or is being taken by yourself.
                First, you should try to recall the objects and places that showed up, and also the actions you took.
                Then, you should raise the question based on the video together with the objects, places and actions. 
                This kind of question should follow this form:
                    “Question: You are moving following a series of movement instructions: "<movement instructions>". What is your current move?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question. The correct answer of this kind of question should contain the exact movement that you are currently performing, which can be extracted from the movement instructions given to you in textual form. The correct answer of this kind of question should like this, with first person perspective:
                “I fly over the buildings.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, also extracted from the movement instructions given to you, but they are not the exact move you are currently taking shown in the video, and the movements or the objects involved can be changed, or made-up. Also remember to use first person perspective.
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are moving following a series of movement instructions: "Turn back. Get down to the chairs between the buildings. Fly over the buildings. Go over the water. Turn right at building." What is your current move?
                    Choices:
                    A.	I turn back.
                    B.	I get down to the chairs between the buildings.
                    C.	I fly over the buildings.
                    D.	I go over the water.
                    E.	I turn right at building.
                    Answer: C
                ”””
            """
        case "Landmark Position":
            category_prompt =  f"""
                To generate a Landmark Position, let's begin thinking step by step.
                First, you should try to recall the objects and places that you saw, and also the actions you took.
                Then, you should raise the question based on the video together with the actions. 
                Then, choose one object or place shown in the video, and use this [certian object] to generate the question, for example, "the intersection beside the tall building".
                This kind of question should follow this form:
                    “Question: You are moving following a series of movement instructions: "<movement instructions>". What is the positional relationship between you and [certain object] when it reached current position?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question. The correct answer should correctly describe your positional relationship with the [certain object]. The correct answer of this kind of question should like this:
                “I am located above the intersection beside the tall building.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the positional description should be twisted, so that these options are contrary to the facts shown in the video. For example, a proper wrong answer might be: “I am located to the right of the intersection beside the tall building.” The description involved can be changed, or made-up.
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are moving following a series of movement instructions: "Turn back. Get down to the chairs between the buildings. Fly over the buildings. Go over the water. Turn right at building." What is the positional relationship between you and the tall building when it reached current position?
                    Choices:
                    A.	I am located under the intersection beside the tall building.
                    B.	I am located above the intersection beside the tall building.
                    C.	I am located to the left of the intersection beside the tall building.
                    D.	I am located on the ground at the intersection beside the tall building. 
                    E.	I am located far away from the intersection beside the tall building.
                    Answer: B
                ”””
            """
        case "Action Generation":
            category_prompt =  f"""
                To generate a Action Generation question, let's begin thinking step by step.
                First, you can refer to your knowledge about the surroundings and also what you've seen along your past moves in the video.
                Then, you should figure out what step is being taken, or is to be taken from what you saw as well as the movement instructions given to you.
                Next, you should raise the question which should exactly follow this form:
                    “Question: You are moving following a series of movement instructions:"<movement instructions>". For your next movement, what is your best next action?” 
                Then, you should generate the options for the question. Every option should contain one of the basic action, or should be a combination of several basic actions. The basic actions are namingly: Ascend, Descend, Fly forward, Turn left, Turn right, Rotate the camera upward, Rotate the camera downward. 
                You should generate ONE correct answer, which you can take to complete the next movement.
                Also, you need to generate FOUR wrong choices for the multi-choice question, which share the same form of the correct option, but they are not the proper action you should take to complete the next movement. The movements involved can be changed, or made-up.
                Finally, you should generate the label of correct answer, from "A" to "E".
                The whole question should be like this:
                “““
                    Question: You are moving following a series of movement instructions: "Turn back. Get down to the chairs between the buildings. Fly over the buildings. Go over the water. Turn right at building." For your next movement, what is your best next action?
                    Choices:
                    A.	I ascend and then turn right.
                    B.	I descend. 
                    C.	I fly forward while descend.
                    D.	I turn left.
                    E.  I rotate the camera upward.
                    Answer: C
                ”””
            """
        case _:
            category_prompt =  "Invalid category, the model must return 'Question Generation Failure because of invalid category'."

    prompt = MAIN_PROMPT + category_prompt + NOTICE
    return prompt


if __name__ == '__main__':

    model = "gemini-1.5-flash"

    import google.generativeai as genai

    genai.configure(api_key="SET_YOUR_API_KEY_HERE")
    

    video_path = rf"PATH\TO\YOUR\VIDEO"  # Replace with your video path. This path should contain video files with video_list.json.
    video_list_path = os.join.path(video_path, "video_list.json") # Video list should contain two columns of "video_name" and "movement_instructions".
    
    
    MCQ_PATH = rf"PATH\TO\YOUR\MCQ\FILE"  # Replace with your MCQ path. This path should contain the generated MCQ, or where you want the MCQ to be saved.
    # Open the MCQ file if it exists, otherwise create a new one.
    if os.path.exists(MCQ_PATH):
        QA_df = pd.read_csv(MCQ_PATH, index_col=0)
    else:
        QA_df = pd.DataFrame(columns=["video_id", "instructions", "question_category", "question", "extracted_answer"])
        QA_df.to_csv(MCQ_PATH, index=True)


    # Gemini
    # Uploaded videos will be saved in a pkl file, which will be used to check if the video has been uploaded to cloud.
    # Saving videos to cloud will help accelerate the whole generation process, as you don't need to upload the same video again and again.
    # The pkl file will be deleted every 8 videos, so that the cloud storage won't be full.
    upload_vid_list_path = 'upload_vid_list.pkl'
    if os.path.exists(upload_vid_list_path):
        with open(upload_vid_list_path, 'rb') as file:
            upload_vid_list = pickle.load(file)
    else:
        upload_vid_list = {}


    # Read the video list
    with open(video_list_path, 'r', encoding='utf-8') as file:
        video_list = json.load(file)

    # Get the video files
    for idx, video_info in enumerate(video_list[:], start=0):
        video_name = video_info["video_name"]
        movement_instructions = video_info["movement_instructions"]

        video_file_name = video_name
        # Check whether the video file is uploaded to cloud or not
        try:
            if video_file_name not in upload_vid_list.keys():
                print(f"Uploading file", video_file_name)
                video_file = genai.upload_file(path=os.path.join(video_path, video_file_name))
                print(f"Completed upload: {video_file.uri}")

                while video_file.state.name == "PROCESSING":
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError(video_file.state.name)

                upload_vid_list[video_file_name] = video_file

                with open(upload_vid_list_path, 'wb') as file:
                    pickle.dump(upload_vid_list, file)

            else:
                video_file = upload_vid_list[video_file_name]

            for question_category in question_categories[0:3]:
                
                # Make the prompt
                prompt = make_prompt(question_category, movement_instructions)

                # Call the model to generate the question and the choices
                client = genai.GenerativeModel(model_name=model)
                response = client.generate_content([video_file, prompt],
                                                    request_options={"timeout": 60})
                content = response.text
                
                # extract the answer given by the model from the response 
                match = re.search(r'Answer:\s*([A-E])', content)
                if match:
                    extracted_answer = match.group(1)
                    # print(f"Extracted Answer: {extracted_answer}")
                else:
                    extracted_answer = "Not Found"
                    print("No answer found")
                content = re.sub(r'Answer:.*', '', content)

                # New entry to be added to the MCQ file
                new_entry = {
                    "video_id": video_file_name,
                    "instructions": movement_instructions,
                    "question_category": question_category,
                    "question": content,
                    "extracted_answer": extracted_answer
                }
                print(new_entry)
                new_entry_df = pd.DataFrame([new_entry], index=[QA_df.index.max() + 1 if not QA_df.empty else 0])
                QA_df = pd.concat([QA_df, new_entry_df], ignore_index=False)
                QA_df.to_csv(MCQ_PATH, index=True)
                QA_df.to_excel(MCQ_PATH.replace(".csv", ".xlsx"), index=True)
                time.sleep(5)
        except Exception as e:
            print(f"Error occured: {e}")
            with open(r"\PATH\TO\ERROR\LOG\FILE", "a", encoding="utf-8") as f:
                f.write(f"Error occurred when processing video: {video_file_name}, Error: {e}\n")
            time.sleep(10)

        if idx % 8 == 0:
            for f in genai.list_files():
                print("  ", f.name)
                f.delete()
            os.remove(upload_vid_list_path)



