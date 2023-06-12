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
    if not hand:
        dictionary = turn_to_default(dictionary)
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

def turn_to_default(dictionary):
    for i in range(26,34):
        dictionary[f'landmark{i}']['x'] = defualt[f'x{i}']
        dictionary[f'landmark{i}']['y'] = defualt[f'y{i}']
        dictionary[f'landmark{i}']['z'] = defualt[f'z{i}']
        dictionary[f'landmark{i}']['visibility'] = defualt[f'v{i}']
    return dictionary

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

defualt = {
"x26": 0.06278683105713041,"y26":0.12439880457711337,"z26": -0.0015778212278352386,"v26": 0.07461923465420509,"x27": 0.05955295160119169,"y27": 0.12278503840192785,"z27": -0.000903746582758157 ,"v27": 0.061697745136756806 ,"x28": 0.061849011044050085 ,"y28": 0.12558278290949432 ,"z28": -0.0017453313579509654 ,"v28": 0.009524793764744296 ,"x29": 0.0630904498321963 ,"y29": 0.1273734431181635 ,"z29": -0.0023157116298521234 ,"v29": 0.008926002346124888 ,"x30": 0.06396057601706148 ,"y30": 0.12869899136385896 ,"z30": -0.0026392119821027197 ,"v30": 0.00870655775134394 ,"x31": 0.058657731897578454,"y31": 0.12412659596339823,"z31": -0.0011870809273942504,"v31": 0.009572297510362688,"x32": 0.06109169839417993,"y32": 0.12708984462204825,"z32": -0.0018519101853769092,"v32": 0.006581293993855661,"x33": 0.06236603334913113,"y33": 0.12877805509003512,"z33": -0.0022159755385966376,"v33": 0.009339161608419726 
}
with open('mobile.pkl', 'rb') as f:
    model = pickle.load(f)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)