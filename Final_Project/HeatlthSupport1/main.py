import google.generativeai as genai
from paddleocr import PaddleOCR
from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt # plot images
import cv2


# flask app
app = Flask(__name__)


# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")


# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))
disease_dict = pickle.load(open('models/ir_updated_disease_info.pkl','rb'))

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def search_disease_in_pickle(symptom):
    if symptom.lower() in disease_dict:
        details = disease_dict[symptom.lower()]
        desc = details["Description"]
        pre = details["Precaution"]
        med = details["Medication"]
        die = details["Diet"]
        wrkout = details["Workout"]
        disease = details["Disease"]
        return desc, pre, med, die, wrkout,disease
    else:
        return None

def add_underscores(sentence):
    words = sentence.split()
    sentence_with_underscores = "_".join(words)
    return sentence_with_underscores

# Function to extract diseases from a sentence using a list of symptoms
def extract_diseases_from_sentence(sentence, symptoms_dict,symptoms_dict_new):
    # Convert the text to lowercase for case-insensitive matching
    text_lower = sentence.lower()
    # List to store found diseases
    diseases = []
    # Iterate through each symptom in the symptoms dictionary
    for symptom in symptoms_dict_new.keys():
        if symptom in text_lower:
            diseases.append(symptom)
            return diseases
    diseases=[]
    for symptom in symptoms_dict.keys():
        if symptom in text_lower:
            diseases.append(symptom)
    return diseases


# creating routes========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/getmedicineAdv', methods=['POST'])
def get_medicine_advantage():
    # Get the data sent from the frontend (button text)
    data = request.json
    print(data)

    # Extract the button text from the data
    button_text = data.get('buttonText', '')
    print(button_text)
    # Creating the string variable
    string_variable = "give me five advantages or disadvantages of " + button_text + " in short"

    # Printing the string variable
    print(string_variable)

    genai.configure(api_key="AIzaSyBrjUnsZTUzQAd6A5eq_cO3KW_wwiWseLk")

    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(string_variable)
    print(response)
    # Splitting the response text into advantages and disadvantages sections
    advantages_section, disadvantages_section = response.text.split("**Disadvantages:**")

    # Cleaning the advantages section
    advantages_section = advantages_section.replace("**Advantages:**", "").strip()

    # Cleaning the disadvantages section
    disadvantages_section = disadvantages_section.strip()

    # Splitting the cleaned sections into lists
    advantages_list = [advantage.strip('* ') for advantage in advantages_section.split('\n') if
                       advantage.strip()]
    disadvantages_list = [disadvantage.strip('* ') for disadvantage in disadvantages_section.split('\n') if
                          disadvantage.strip()]

    # Printing the cleaned lists
    print("Advantages:")
    print(advantages_list)
    print("\nDisadvantages:")
    print(disadvantages_list)

    response_data = {'advantages': advantages_list, 'disadvantages': disadvantages_list}
    return jsonify(response_data)

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                # Save the uploaded image with a constant name in the 'static' folder
                image.save('static/uploaded_image.jpg')
                ocr_model = PaddleOCR(use_gpu=False, lang='en')
                result = ocr_model.ocr("static/uploaded_image.jpg")
                print(result)
                # Iterate through the nested list and print each element
                # List to store tuples
                tuples_list = []

                # Iterate through the nested list and store tuples
                for level1 in result:
                    for level2 in level1:
                        for level3 in level2:
                            if isinstance(level3, tuple):
                                tuples_list.append(level3)

                # Print the list containing tuples
                # Print the list containing tuples before sorting
                print("Before sorting:")
                print(tuples_list)

                # Sort the list of tuples based on the second element of each tuple in reverse order
                sorted_tuples_list = sorted(tuples_list, key=lambda x: x[1], reverse=True)

                # Print the sorted list of tuples
                print("After sorting:")
                print(sorted_tuples_list)

                print(sorted_tuples_list[0][0])
                drug_name = sorted_tuples_list[0][0]

                # Creating the string variable
                string_variable = "give me five advantages or disadvantages of " + drug_name + " in short"

                # Printing the string variable
                print(string_variable)

                genai.configure(api_key="AIzaSyBrjUnsZTUzQAd6A5eq_cO3KW_wwiWseLk")

                model = genai.GenerativeModel('gemini-pro')

                response = model.generate_content(string_variable)
                print(response)

                # Splitting the response text into advantages and disadvantages sections
                advantages_section, disadvantages_section = response.text.split("**Disadvantages:**")

                # Cleaning the advantages section
                advantages_section = advantages_section.replace("**Advantages:**", "").strip()

                # Cleaning the disadvantages section
                disadvantages_section = disadvantages_section.strip()

                # Splitting the cleaned sections into lists
                advantages_list = [advantage.strip('* ') for advantage in advantages_section.split('\n') if
                                   advantage.strip()]
                disadvantages_list = [disadvantage.strip('* ') for disadvantage in disadvantages_section.split('\n') if
                                      disadvantage.strip()]

                # Printing the cleaned lists
                print("Advantages:")
                print(advantages_list)
                print("\nDisadvantages:")
                print(disadvantages_list)
                # Generate a random paragraph for image upload
                random_paragraph = "asdjsd fcs dfjsd"
                return render_template('index.html', advres=advantages_list , disres=disadvantages_list)

        # Check if symptoms were provided
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = symptoms
            sentence_new = add_underscores(user_symptoms)
            user_symptoms = extract_diseases_from_sentence(sentence_new, symptoms_dict, disease_dict)
            print(user_symptoms)

            # Remove any extra characters, if any
            # user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            if isinstance(user_symptoms, list) and len(user_symptoms) == 1:
                symptom = user_symptoms[0]
                if search_disease_in_pickle(symptom) != None:
                    dis_des, my_precautions, medications, rec_diet, workout, predicted_disease = search_disease_in_pickle(
                        symptom)
                    print(my_precautions)
                else:
                    predicted_disease = get_predicted_value(user_symptoms)
                    dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
                    my_precautions = []
                    for i in precautions[0]:
                        my_precautions.append(i)
            else:
                predicted_disease = get_predicted_value(user_symptoms)
                dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
                my_precautions = []
                for i in precautions[0]:
                    my_precautions.append(i)


            # predicted_disease = get_predicted_value(user_symptoms)
            # dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            # medications = generate_content(medications)
            print(workout)
            print(medications)
            print(my_precautions)

            # Function to check if the string is in the desired format
            def is_desired_format(item):
                return isinstance(item, str) and item.startswith("['") and item.endswith("']")

            # Check if all items in the first list are in the desired format
            if all(is_desired_format(item) for item in medications):
                # Extract medications from each item in the first list
                medications_list = [item.strip("['").strip("']").split("', '") for item in medications][0]
            else:
                medications_list = medications

            print(medications_list)
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')





# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")





if __name__ == '__main__':
   app.run(debug=True)

