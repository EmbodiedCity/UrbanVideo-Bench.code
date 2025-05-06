import time
import os
import re
import pandas as pd
import pickle
import json

question_categories = [
    "Captioning",   
    "Start/End",    
    "Object Recall",    
    "Sequence Recall",
    "Scene Recall", 
    "Proximity",
    "Duration",
    "Causal",
    "Counterfactual"
]

def make_prompt_move(destination):
    MAIN_PROMPT = f"""
            MAIN INSTRUCTIONS:
            In the "Egocentric UAV Video Narration" task, you are working with a choronological series of frames from an egocentric video taken by a UAV while it flies towards a specific known destination in a city scene.
            Your goal is to write a chronological Narration of the route of the UAV that are shown in the video frames. 

            For the video Narration, you should first show the [starting position] and the [final destination] of the UAV. 
            Then you chronologically list the specific [movements] and [positions] according to the video frames. 
            The [movements] may include but not limited to "goes forward to", "turns left to face", "descend to the height of", etc. 
            The [positions] may include "the center of the square", "the balcony on the 10th floor", "left of main entrance", etc.
            The [destination] is known, given by textual input.

            The video frame may contain movements taken on the way to the destination, but it is possible that the destination is not reached in the video frame.
            For each or each several frames, you should only describe the movements and positions of the UAV shown in the video frame.

            EXAMPLE OUTPUT:
            {
                "In the video, the UAV goes from in front of the ""北京烟草"" building, to the main entrance."
                "Movement 1: The UAV descends to the height of the 1st floor."
                "Movement 2: The UAV turns downward to face the main entrance."
                "............................."
                "Movement X: The UAV goes forward to the main entrance."
            }

            EXAMPLE explain:
            The first line summarizes the starting position and the destination of the UAV.
            Each of the following lines contains a single, detailed movement of the UAV, Starting with "Movement X:".
            The movements are in chronological order, and each movement should be a detailed phrase describing the movement or position of the UAV in the video, so that there'd be no confusion.

            NOTICE: Extra explanation of the output is not allowed. Your output should be a list of phrases ONLY. When you try to describe directions or positional relationship, you should use left, right, front, back, etc, instead of north, south, east, west.

            The inputs are as follows:
            The destination of the UAV is: {destination}

    """

    prompt = MAIN_PROMPT
    return prompt

def make_prompt_object(instructions):
    MAIN_PROMPT = f"""
            MAIN INSTRUCTIONS:
            In the "Egocentric UAV Video Scene Recall and Object Extract" task, you are working with a choronological series of frames from an egocentric video taken by a UAV while it flies in a city scene.
            Your goal is to write a chronological list of the objects and positions the UAV came across along its route shown in the video frames. 

            You should give the list in a chronological order, and each item should be a detailed phrase describing the object or position the UAV came across in the video frame, so that there'd be no confusion.
            Especially, you should put the starting and ending position of the UAV in the list, on top and bottom respectively.
            An simple series of instructions are given to you in textual form, which you can refer to when writing the list.

            The output should be a list of phrases, following the format below:

            EXAMPLE OUTPUT:
            {
                "The building beside the square", "The window to the right of the main gate", "The red car parked on the side of the road", "The traffic light at the closest intersection"
            }
            
            NOTICE: Extra explanation of the output is not allowed. Your output should be a list of phrases ONLY. When you try to describe directions or positional relationship, you should use left, right, front, back, etc, instead of north, south, east, west.

            The INPUTS are as follows:
            The instructions along the route are: {instructions}
    """

    prompt = MAIN_PROMPT
    return prompt

def make_prompt(question_category, destination, movements, object_list):
    MAIN_PROMPT = f"""
            MAIN INSTRUCTIONS:
            You are the teacher of the course "Video comprehension and Spatial reasoning". To test the students, you need to make many test questions from a series of egocentric videos taken by a UAV while it flies towards a specific known destination in a city scene.
            Your goal is to raise multi choice questions about the details and spatial or temporal logic from video that satisfy later-on given requirements.
            
            For each video, you must raise a multi choice question, that must be strictly restricted to and strongly related to the video content. 
            For each question, you need to raise the question itself, and then give 5 choices, labeled as A, B, C, D and E, including ONLY 1 CORRECT answer and 4 wrong answers. Specially, the "Proximity", "Duration" and "Counterfactual" questions should only contain 3 choices, including 1 correct answer and 2 wrong answers.
            For the choices, you need to make sure that the correct answer is not too obvious, and the wrong answers are not too irrelevant. 
            You are allowed to put the correct answer at any position among the 5 choices.
            Finally, you need to give the correct answer to the question, which is one of A, B, C, D and E.

            All possible questions falls into these categories, that covers different aspects of the student's ability to understand the video content, spatial and temporal relationship and causal logic.
            
            The categories, templates and examples of Question, Choices and Answer are given respectively as follows:
            1.  Captioning
                TASK EXPLANATION:   This question requires the student to summarize your movement route in the video by combining the specific objects along the way. Proper answer should be a concise summary of your movement route, including its starting location, final destination and specific buildings, objects along the way.
                TEMPLATE Question:  Summarize your [movement route] in the video by combining the [specific objects] along the way.
                TEMPLATE Choices:   I go from [specific objects] to the [specific objects] by [specific movenments about specific objects].

                EXAMPLE Question:   According to the video, which of the following choices better summarizes your movement route?
                EXAMPLE Choices:    A.  I go from [between two tall skyscrapers] to the [the balcony on the 31st floor] by flying straight across the opening ground below.
                                    B.  I go from [the center of the square] to the [left of the main entrance] by descending height to the rooftop.
                                    C.  I go from [the main entrance] to the [right of the main entrance] by turning left to face the main entrance.
                                    D.  I go from [the balcony on the 10th floor] to the [the center of the square] by descending to the ground level.
                                    E.  I go from  [between two tall skyscrapers] to the [the balcony on the 30th floor] by flying straight across the opening ground below.
                EXAMPLE Answer:     A
            
            2.  Start/End
                TASK EXPLANATION:   This question requires the student to identify your starting and ending points in the video. Proper answer should include the specific locations where you begins and ends its journey.
                TEMPLATE Question:  Where are your starting point and ending point in the video?
                TEMPLATE Choices:   I start at [specific location] and ends at [specific location].

                EXAMPLE Question:   According to the video, where do you start and end its journey?
                EXAMPLE Choices:    A.  I start at [the main entrance] and ends at [the rooftop].
                                    B.  I start at [the center of the square] and ends at [the balcony on the 10th floor].
                                    C.  I start at [the left of the main entrance] and ends at [the center of the square].
                                    D.  I start at [the balcony on the 31st floor] and ends at [the main entrance].
                                    E.  I start at [the rooftop] and ends at [the center of the square].
                EXAMPLE Answer:     B
            
            3.  Object Perception
                TASK EXPLANATION:   This question requires the student to recall specific objects that are present near other specific objects in the video. Proper answer should include the correct identification of the objects and their relative positions.
                TEMPLATE Question:  According to the video, what is [at specific location/direction of specific object]?
                TEMPLATE Choices:   There is a [specific object] near the [specific object].

                EXAMPLE Question:   According to the video, what is on the left outer surface of the building?
                EXAMPLE Choices:    A.  There is a billboard on the left outer surface of the building.
                                    B.  There is a balcony on the left outer surface of the building.
                                    C.  There is a roadlamp on the left outer surface of the building.
                                    D.  There is a car on the left outer surface of the building.
                                    E.  There is nothing on the left outer surface of the building.
                EXAMPLE Answer:     A
            
            4.  Sequence Recall
                TASK EXPLANATION:   This question requires the student to recall your next action after a specific action in the video. Proper answer should include the correct identification of the next action performed by you.
                TEMPLATE Question:  What is your [next action] after the [specific action]?
                TEMPLATE Choices:   I [next action] after the [specific action].

                EXAMPLE Question:   What is your next action after turning left at the intersection?
                EXAMPLE Choices:    A.  I ascend to a higher altitude.
                                    B.  I turn right.
                                    C.  I slowly fly straight ahead.
                                    D.  I descend to a lower altitude.
                                    E.  I hover in place.
                EXAMPLE Answer:     C
            
            5.  Scene Perception  
                TASK EXPLANATION:   This question requires the student to identify the objects observed by you during a specific action in the video. Proper answer should include the correct identification of the objects and their relative positions.
                TEMPLATE Question:  What [objects] do you observe during the [specific action]?
                TEMPLATE Choices:   I observ [specific objects] during the [specific action].

                EXAMPLE Question:   What objects did you observe while flying over the park?
                EXAMPLE Choices:    A.  I observe a bench, a fountain, and a tree while flying over the park.
                                    B.  I observe a car, a streetlight, and a building while flying over the park.
                                    C.  I observe a bicycle, a dog, and a playground while flying over the park.
                                    D.  I observe a statue, a pond, and a flowerbed while flying over the park.
                                    E.  I observe a bridge, a river, and a boat while flying over the park.
                EXAMPLE Answer:     A
            
            6.  Proximity
                TASK EXPLANATION:   This question requires the student to identify how the distance between you and a specific object changes after a specific action in the video. Proper answer should describe the correct change in distance between you and the specific object after the move.
                TEMPLATE Question:  How does the distance between you and the [specific object] change after the [specific action]?
                TEMPLATE Choices:   The distance between you and the [specific object] [change in distance] after the [specific action].

                EXAMPLE Question:   How does the distance between you and the building change after you move forward?
                EXAMPLE Choices:    A.  The distance between me and the building increases.
                                    B.  The distance between me and the building decreases.
                                    C.  The distance between me and the building remains the same.
                EXAMPLE Answer:     B

            7. Duration
                TASK EXPLANATION:   This question requires the student to compare the duration of two specific actions performed by you in the video. Proper answer should identify which action takes longer.
                TEMPLATE Question:  Which takes longer, your [specific action] or [specific action]?
                TEMPLATE Choices:   My [specific action] takes [longer/shorter] than the [specific action].

                EXAMPLE Question:   Which takes longer, your ascent to the rooftop or its descent to the ground level?
                EXAMPLE Choices:    A.  My ascent to the rooftop takes longer than its descent to the ground level.
                                    B.  My ascent to the rooftop takes shorter than its descent to the ground level.
                                    C.  My ascent to the rooftop takes the same time as its descent to the ground level.
                EXAMPLE Answer:     B
            
            8. Causal
                TASK EXPLANATION:   This question requires the student to identify the reason behind your specific action after another specific action in the video. Proper answer should explain the cause-and-effect relationship between the two actions.
                TEMPLATE Question:  Why do you perform the [specific action] after the [specific action]?
                TEMPLATE Choices:   I perform the [specific action] after the [specific action] because [reason].

                EXAMPLE Question:   Why do you descend after reaching the rooftop?
                EXAMPLE Choices:    A.  I descend after reaching the rooftop because it needed to land.
                                    B.  I descend after reaching the rooftop because it was avoiding an obstacle.
                                    C.  I descend after reaching the rooftop because its earlier field of view is not good for observation.
                                    D.  I descend after reaching the rooftop because it was running low on battery.
                                    E.  I descend after reaching the rooftop because it was capturing a closer view.
                EXAMPLE Answer:     A  
            
            9. Counterfactual
                TASK EXPLANATION:   This question requires the student to consider an alternative scenario where you takes a hypothetical movement route different from a specific movement shown in the video frames, and determine whether it can still complete the task. Proper answer should evaluate the feasibility of the alternative route. 
                TEMPLATE Question:  Instead of taking [specific movement], if you choose [hypothetical movement], can it complete the task, and how is the alternative route?
                TEMPLATE Choices:   If I choose [hypothetical movement], it [can/cannot] complete the task because [reason], [the alternative takes longer/shorter time].

                EXAMPLE Question:   If you chooses to fly around the building instead of over it, can it still reach the main entrance?
                EXAMPLE Choices:    A.  If I chooses to fly around the building, it can complete the task because it avoids obstacles, and the hypothetical movement takes shorter time.
                                    B.  If I chooses to fly around the building, it cannot complete the task because the final destination is near the ground.
                                    C.  If I chooses to fly around the building, it can complete the task because it avoids obstacles, but the hypothetical movement takes longer time.
                EXAMPLE Answer:     B
            
            

            
            NOTICE: 
            The [category] and the [destination] is known, given by textual input.
            Always use second person perspective when generating questions. For example, use “You are moving following a series of movement instructions.” instead of “The UAV is moving following a series of movement instructions.”.
            Always use first person perspective when generating the choices. For example, use “I am flying over the buildings.” instead of “The UAV is flying over the buildings.”.
            Always use present tense when generating the questions and the choices. For example, use “What is your current move?” instead of “What was the UAV's current move?”.
            NEVER add extra explanatory text to the output. The output should only contain the Question, the Choices and the Answer, exactly in the form given above.
            When you fail to generate the question, simply output “Question Generation Failure”.
            When you try to describe directions or positional relationship, you should use left, right, front, back, etc, instead of north, south, east, west.
            ALWAYS remember to generate options that are different from the examples above, and also different from each other.

            EXAMPLE INPUT:
            
            Please raise a [category] multi choice question based on the video taken along the way towards [destination].
            For your reference, the content of the video is shown in a chornological narration, which is given as follows:
            ........(narration text)........
            The frames from the video are as follows:
            ........(video frames)........

            EXAMPLE OUTPUT:
            '''
                Question:   ............................?
                Choices:    A. ..........................
                            B. ..........................
                            C. ..........................
                            D. ..........................
                            E. ..........................
                Answer:     A
            '''
            
            You output must only contain the Question, Choices and Answer in the form shown above, and any other reply, such as "Certainly, here's a .... question about... " is definately NOT ALLOWED!
            The INPUT are as follows:
            Please raise a {question_category} multi choice question based on the video taken along the way towards {destination}.
            For your reference, the content of the video is shown in a chornological narration, which is given as follows:
            {movements}
            Also, the objects and positions mentioned in the questions and correct choices should come from the video frames, or you can refer to the list below:
            {object_list}
            

        """

    prompt = MAIN_PROMPT
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
        QA_df = pd.DataFrame(columns=["video_name", "movements", "objects", "destination", "question_category", "question", "extracted_answer"])
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
    for idx, video_info in enumerate(video_list[:], start=64):
        video_name = video_info["video_name"]
        destination = video_info["destination"]

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

            
            # Extract the movements and objects from the video with MLLM.           
            prompt = make_prompt_move(destination)
            client = genai.GenerativeModel(model_name=model)
            response = client.generate_content([video_file, prompt],
                                                request_options={"timeout": 60})
            movements = response.text
        
            
            prompt = make_prompt_object(movements)
            client = genai.GenerativeModel(model_name=model)
            response = client.generate_content([video_file, prompt],
                                                request_options={"timeout": 60})
            objects = response.text

            # For each video, generate 9 questions, one for each category.
            for question_category in question_categories[0:9]:

                # Make the prompt
                prompt = make_prompt(question_category, destination, movements, objects)

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
                    "video_name": video_file_name,
                    "movements": movements,
                    "objects": objects,
                    "destination": destination,
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



