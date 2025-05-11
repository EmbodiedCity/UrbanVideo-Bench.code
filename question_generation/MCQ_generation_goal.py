import time
import os
import re
import pandas as pd
import pickle
import json

# This script is used for generating multiple-choice questions (MCQs) of Goal Oriented Navigation Tasks for given videos using the Gemini model. For more detail of the Goal Oriented Navigation Tasks, please refer to our paper.

question_categories = [
    "Trajectory Captioning",
    "Landmark Position",
    "Goal Detection",
    "Association",
    "Cognitive Map",
    "High-level Planning",
    "Action Generation"
]

def make_prompt(question_category, video_destination):
    
    MAIN_PROMPT = f"""
    MAIN INSTRUCTIONS:
        You are a marvoulous embodied UAV AI flying in urban areas. Your goal is to recall and comprehend the in-coming video input taken by your camera, and try to think like humans do to control your movement.
        When you think about the environment and navigation, you should raise textual questions around the video input, so as to help your thinking, and also to help the engineers that developed you to understand what you think. Your goal is to raise multi choice questions about the details or logic from video, and then generate proper choices and the correct answer. You must strictly follow the later-on given requirements.
        Now, you are required to raise a {question_category} question based what you've seen. 
        You are navigating to the desitnation of {video_destination}, and you should generate a question based on the video input together with the objects, places and actions.
        For this kind of question, the special requirements are as follows:
    """

    NOTICE = f"""
    Always use second person perspective when generating questions. For example, use “You are moving following a series of movement instructions.” instead of “The UAV is moving following a series of movement instructions.”.
    Always use first person perspective when generating the choices. For example, use “I am flying over the buildings.” instead of “The UAV is flying over the buildings.”.
    Always use present tense when generating the questions and the choices. For example, use “What is the UAV's current move?” instead of “What was the UAV's current move?”.
    NEVER add extra explanatory text to the output. The output should only contain the Question, the Choices and the Answer, exactly in the form given above.
	When you fail to generate the question, simply output “Question Generation Failure”.
    When you try to describe directions or positional relationship, you should use left, right, front, back, etc, instead of north, south, east, west.
    ALWAYS remember to generate options that are different from the examples above, and also different from each other.
    """

    match question_category:
        case "Trajectory Captioning":
            category_prompt =  f"""
                To generate a Trajectory Captioning question, let's begin thinking step by step.
                First, you should try to recall the objects and places that showed up, and also the actions you took.
                Then, you should raise the question based on the video together with the objects, places and actions. 
                This kind of question should follow this form:
                    “Question: You are navigating to [destination], what is the proper summary of your trajectory from the beginning till current position?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question. The correct answer of this kind of question should like this:
                “I start from the pedal lane in the park, turn left to face the street outside the park, rise to the height of the rooftop of the high-rise across the street, and move forward to reach the current position outside the edge of the rooftop of the high-rise.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the details might be faked or twisted or the important information should be hidden, so that these options are contrary to the facts shown in the video.
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [the rooftop of the high-rise across the street from the park], what is the proper summary of your trajectory from the beginning till current position?
                    Choices:
                    A.	I start from the pedal lane in the park, move forward to reach the street, and turn left to face the lamppost.
                    B.	I start from the pedal lane in the park, turn left to face the street outside the park, rise to the height of the rooftop of the high-rise across the street, and move forward to reach the centre of the rooftop of the high-rise.
                    C.	I start from the street, move along the pedestrian way, and reach the gate of the park.
                    D.	I start from the pedal lane in the park, turn left to face the street outside the park, rise to the height of the rooftop of the high-rise across the street, and move forward to reach the current position outside the edge of the rooftop of the high-rise.
                    E.	I rise from the park, and descend to reach the edge of the rooftop of the high-rise.
                    Answer: D
                ”””
            """
        case "Landmark Position":
            category_prompt =  f"""
                To generate a Landmark Position question, let's begin thinking step by step.
                First, you should try to recall the objects and places that showed up, and also the actions you took.
                Then, you should raise the question based on the video together with the actions. 
                This kind of question should follow this form:
                    “Question: You are navigating to [destination], what is the Direction between you and the destination when it reached current position?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question. The correct answer should correctly describe the UAV's Direction with the destination. The correct answer of this kind of question should like this:
                “You are currently by the edge of the rooftop of the high-rise across the street from the park.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the positional description should be twisted, so that these options are contrary to the facts shown in the video. For example, a proper wrong answer might be: “You are currently right above the centre of the rooftop of the high-rise across the street from the park.”
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [the rooftop of the high-rise across the street from the park], what is the positional relationship between you and the destination when it reached current position?
                    Choices:
                    A.	You are currently by the edge of the rooftop of the high-rise across the street from the park.
                    B.	You are currently right above the centre of the rooftop of the high-rise across the street from the park.
                    C.	You are currently close to the side facade of the high-rise across the street from the park, at height of the middle levels.
                    D.	You are currently close to the side facade of the high-rise across the street from the park, at height of the bottom levels. 
                    E.	You are currently above the park lane, much higher than the high-rise.
                    Answer: A
                ”””
            """
        case "Goal Detection":
            category_prompt =  f"""
                To generate a Goal Detection question, let's begin thinking step by step.
                First, you should try to recall the objects and places that showed up.
                Then, you should raise the question based on the video together with the objects or places. 
                This kind of question should follow this form:
                    “Question: You are navigating to [destination]. At current position, is the destination in your view? If yes, where is the destination in your view?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question. The correct option should first answer whether the destination is in the view, and if yes, then correctly describe the destination's location in the view, which might be one of “in the centre”, “on the left”, “on the top right corner”, etc. 
                If the destination is in the view, the correct answer of this kind of question should like this:
                “The rooftop of the high-rise across the street from the park is currently in the centre of my view.”
                If the destination is not in the view, the answer should be like this:
                    “The rooftop of the high-rise across the street from the park is not in my view.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the positional description should be twisted, or the existence should be modified, so that these options are contrary to the facts shown in the video.
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [the rooftop of the high-rise across the street from the park]. At current position, is the destination in your view? If yes, where is the destination in your view?
                    Choices:
                    A.	The rooftop of the high-rise across the street from the park is not in my view.
                    B.	The rooftop of the high-rise across the street from the park is currently in the left part of my view. 
                    C.	The rooftop of the high-rise across the street from the park is currently in the top left corner of my view. 
                    D.	The rooftop of the high-rise across the street from the park is currently in the bottom right of my view. 
                    E.	The rooftop of the high-rise across the street from the park is currently in the centre of my view.
                    Answer: E
                ”””
            """
        case "Association":
            category_prompt =  f"""
                To generate an Association question, let's begin thinking step by step.
                First, you should try to recall the objects and places that showed up.
                Next, you should raise the question which should exactly follow this form:
                    “Question: You are navigating to [The rooftop of the high-rise across the street]. At current position, if the [The rooftop of the high-rise across the street] is not in your field of view, what is the most relative object in sight, and where is this object?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question, based on the video together with the objects and places. 
                If the destination is already in view, the correct option should be: 
                    “[The rooftop of the high-rise across the street] is already in my view.” 
                If the destination is currently not in view, the correct option should tell one of the objects or places that is most likely relative to the destination, for example such option should be like: 
                    “[The rooftop of the high-rise across the street] is not in my view, but there is the main gate of a high-rise across the street in the left part of my view.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the positional description and related objects should be twisted, so that these options are contrary to the facts shown in the video. 
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [The rooftop of the high-rise across the street]. At current position, if the [The rooftop of the high-rise across the street] is not in your field of view, what is the most relative object in sight, and where is this object?
                    Choices:
                    A.	[The rooftop of the high-rise across the street] is already in my view.
                    B.	[The rooftop of the high-rise across the street] is not in my view, but there is the facade of a low-level building seen in the left part of my view. 
                    C.	[The rooftop of the high-rise across the street] is not in my view, but there is the main gate of a high-rise across the street in the left part of my view. 
                    D.	[The rooftop of the high-rise across the street] is not in my view, but a bus stop is in the lower part of my view. 
                    E.	[The rooftop of the high-rise across the street] is not in my view, but the lake area in the park is in the bottom right corner of my view.
                    Answer: C
                ”””
            """
        case "Cognitive Map":
            category_prompt =  f"""
                To generate an Cognitive Map question, let's begin thinking step by step.
                First, you should try to recall the objects and places that showed up, and also the actions you took.
                Next, you should raise the question which should exactly follow this form:
                    “Question: You are navigating to [destination]. Recalling the past actions and the objects seen in the video, what is the surrounding like at your current position?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question, based on the video together with the objects and actions. The correct option should contain detailed description of the buildings, objects and places around the UAV that are extracted from the video, and also their Direction with the UAV. The correct option should be like this:
                    “There are three buildings lined in a row across the street in front of me, the middle one is much taller than the other two. There's a park with pedal lanes and trees behind me. There is a pedestrian way right beneath me.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, containing surrounding objects and special relationship, but the positional description and related objects should be changed or faked, so that these options are contrary to the facts shown in the video. 
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [The rooftop of the high-rise across the street]. Recalling the past actions and the objects seen in the video, what is the surrounding like at your current position?
                    Choices:
                    A.	There's one flat building in front of me. There's a park with playground behind me. There is a pedestrian way right beneath me. 
                    B.	There's three buildings lined in a row across the street in front of me, the middle one is much taller than the other two. There's a park with pedal lanes and trees behind me. There is a pedestrian way right beneath me. 
                    C.	There's a train rail in front of me along the street. There's a train station on the left side of me.
                    D.	There's three buildings lined in a row across the street in front of me, the middle one is much shorter than the other two. There's a square behind me. 
                    E.	There's four buildings lined in a row across the street in front of me. There's a park with pedal lanes and trees behind me. There is a pedestrian way right beneath me. 
                    Answer: B
                ”””
            """
        case "High-level Planning":
            category_prompt =  f"""
                To generate a High-level Planning question, let's begin thinking step by step.
                First, you should recall your knowledge about the surroundings, and what you've seen along your past moves in the video.
                Next, you should raise the question which should exactly follow this form:
                    “Question: You are navigating to [destination]. Recalling the past actions and the objects seen in the video, in order to reach the destination from current position, what should you approach next?”
                Then, you should generate ONE correct answer as the ONLY correct choice of the question, based on the video together with the objects and actions. The correct option should be a proper location for the UAV to reach next when it continues its navigation from its current position to the final destiny. The correct option should be like this:
                    “The height of the rooftop of the high-rise.”
                Also, you need to generate FOUR wrong answers for the multi-choice question. The wrong answers should share the same form of the correct one, but the position or details of the objects mentioned should be changed or faked, so that these options are not a proper milestone for the navigation. 
                Finally, you should RANDOMLY reorder the five options and give them alphabetical labels from A to E. You must remember the label of the correct answer, so that the engineers will be sure that your process of thinking is correct.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [The rooftop of the high-rise across the street]. Recalling the past actions and the objects seen in the video, in order to reach the destination from current position, what should you approach next?
                    Choices:
                    A.	The canopy of the nearest tree along the street.
                    B.	The mailbox by the entrance of the building. 
                    C.	The intersection at the gate of the park.
                    D.	The height of the rooftop of the high-rise. 
                    E.	The bench along the lane in the park. 
                    Answer: D
                ”””
            """
        case "Action Generation":
            category_prompt =  f"""
                To generate a Action Generation question, let's begin thinking step by step.
                First, you are given the latest movement plan: [planned milestone], and you can refer to your knowledge about the surroundings and also what you've seen along your past moves in the video.
                Next, you should raise the question which should exactly follow this form:
                    “Question: You are navigating to [destination]. Considering the next movement plan, what Actions should you perform next?”
                Then, you should generate the options for the question. There should be exactly 7 options from A to G, namingly:
                    A.	Ascend.
                    B.	Descend.
                    C.	Fly forward.
                    D.	Turn left.
                    E.	Turn right.
                    F.	Rotate the camera upward.
                    G.	Rotate the camera downward.
                The correct answer should be one of the options from A to G, which is the proper action for the UAV to take next when it continues its navigation from its current position to the final destiny.
                At last, output the question that you've got in textual form. The whole question should be like this:
                “““
                    Question: You are navigating to [the rooftop of the high-rise across the street]. Given your next movement plan, what Actions should you perform next??
                    Choices:
                    A.	Ascend.
                    B.	Descend. 
                    C.	Fly forward.
                    D.	Turn left.
                    E.  Turn right. 
                    F.	Rotate the camera upward.
                    G.	Rotate the camera downward. 
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
    video_list_path = os.join.path(video_path, "video_list.json") # Video list should contain two columns of "video_name" and "destination".
    
    
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
        video_destination = video_info["destination"]
        
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

            for question_category in question_categories[0:7]:
                
                # Make the prompt
                prompt = make_prompt(question_category, video_destination)

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
                    "destiny": video_destination,
                    "question_category": question_category,
                    "question": content,
                    "extracted_answer": extracted_answer
                }
                print(new_entry)
                new_entry_df = pd.DataFrame([new_entry], index=[QA_df.index.max() + 1 if not QA_df.empty else 0])
                QA_df = pd.concat([QA_df, new_entry_df], ignore_index=False)
                QA_df.to_csv(MCQ_PATH, index=True)
                QA_df.to_excel(MCQ_PATH.replace(".csv", ".xlsx"), index=True)
                QA_df.to_json(MCQ_PATH.replace(".csv", ".json"), orient='records', lines=True, force_ascii=False)
                time.sleep(5)
        except Exception as e:
            print(f"Error occured: {e}")
            with open(r"error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Error occurred when processing video: {video_file_name}, Error: {e}\n")
            time.sleep(10)
        
        if idx % 8 == 0:
            for f in genai.list_files():
                print("  ", f.name)
                f.delete()
            os.remove(upload_vid_list_path)



