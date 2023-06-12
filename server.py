from flask import Flask, request
import numpy as np
import pandas as pd
import json
import pickle
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def receive_json():
    json_data = request.get_json()
    # Process the received JSON data
    # You can access the data using json_data['key']
    
    # Example: Print the received data
    pose = convertLandmarkToPython(json_data["pose"],False)
    left = None
    right = None
    try:
        left = convertLandmarkToPython(json_data["left"],True)
        print("left")
        # print(left)
    except Exception as e:
        print("left error")
        print(e)
    try:
        right = convertLandmarkToPython(json_data["right"],True)
        print("right")
        # print(right)
    except Exception as e:
        print("right error")
        print(e)
    
    prediction = predict(pose,left,right)
    print(prediction)
    return prediction


def convertLandmarkToPython(content,hand):
    modified_string = delete_first_line(content)  
    modified_string = replace_with_number(modified_string,'landmark')
    modified_string = modified_string.strip().replace(' ', '')
    if not hand:
        modified_string = modified_string.replace('presence:', '"presence":')
        modified_string = modified_string.replace('visibility:', ',"visibility":')
        modified_string = modified_string.replace('x:', ',"x":')
    modified_string = modified_string.replace('x:', '"x":')
    modified_string = modified_string.replace('y:', ',"y":')
    modified_string = modified_string.replace('z:', ',"z":')

    modified_string = modified_string.replace('}', '},') + modified_string[1]
    modified_string = modified_string.rsplit(',', 1)[0]

    dictionary = json.loads('{' + modified_string + '}')
    return dictionary

def predict(pose,left,right):
    pose_list = list(np.array([[pose[res]["x"], pose[res]["y"],pose[res]["z"], pose[res]["visibility"]] for res in pose]).flatten(
    ) if pose else np.zeros(33*4))
    left_list = list(np.array([[left[res]["x"], left[res]["y"],left[res]["z"]] for res in left]).flatten() if left else np.zeros(21*3))
    right_list = list(np.array([[right[res]["x"], right[res]["y"],right[res]["z"]] for res in right]).flatten() if right else np.zeros(21*3))                
    # Concate rows
    row = pose_list +left_list+right_list
    X = pd.DataFrame([row])
    Sign_language_class = model.predict(X)[0]
    Sign_language_prob = model.predict_proba(X)[0]
    return Sign_language_class.split(' ')[0]


def replace_with_number(sentence, word_to_replace):
    count = 1
    words = sentence.split()
    replaced_sentence = ''
    for word in words:
        if word == word_to_replace:
            replaced_sentence += f'"{word}{count}":'
            count += 1
        else:
            replaced_sentence += f'{word} '
    return replaced_sentence.strip()

def delete_first_line(text):
    lines = text.splitlines()
    if len(lines) > 1:
        lines = lines[1:]
    return '\n'.join(lines)


with open('ETHSLv4.pkl', 'rb') as f:
    model = pickle.load(f)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)