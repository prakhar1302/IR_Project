# MedicalSupport
## Problem Statement
Manual research for medical information and treatment options can be time-consuming and complex, particularly for individuals with limited medical knowledge. Accessing relevant medical information efficiently is crucial for making informed decisions about healthcare. An automated solution can streamline this process and provide accessible support.
Finding medical information and treatment options can be hard, especially if you're not a medical expert. Getting the right info quickly is super important for making good healthcare choices. Using technology to help can make this whole process much easier
## Proposed Solution
Our proposed solution for health support system begins with users inputting their symptoms using free-text, where advanced Natural Language Processing (NLP) techniques, specifically Named Entity Recognition (NER), are employed to extract specific symptoms. The dataset undergoes rigorous cleaning to ensure accuracy, followed by mapping diseases with their corresponding symptoms. This meticulous process lays the foundation for precise disease prediction.

The heart of our system lies in the application of the Support Vector Classifier (SVC) algorithm. Trained on the cleaned and mapped dataset, the SVC model learns intricate patterns and relationships between symptoms and diseases. Once trained, the model is capable of accurately predicting diseases based on the input symptoms provided by the user, enhancing diagnostic capabilities significantly.

Upon successful disease prediction, our system generates comprehensive recommendations tailored to each user. These recommendations encompass a range of aspects, including disease diagnosis, personalized diet plans, workout routines, medication suggestions, and precautionary measures. This holistic approach ensures that users receive a well-rounded health support system that addresses their unique needs.

Furthermore, our system integrates Gemini Pro AI for in-depth medication analysis, delving into the advantages and disadvantages of various medications. This analysis considers both the medication data and the symptoms/diseases predicted by the model, providing users with valuable insights into their prescribed medications.

To further enhance medication recommendations, we incorporate Optical Character Recognition (OCR) technology. By extracting information from medication labels uploaded by users, our system gains a deeper understanding of medication specifics, enabling us to offer detailed recommendations that include medication details along with their advantages and disadvantages, as well as additional precautions.

In conclusion, our health support system encompasses a comprehensive approach, from symptom extraction to disease prediction, personalized recommendations, medication analysis, and OCR-based insights, ultimately providing users with a comprehensive and personalized health management solution.
write the above 
 ## Result 
 ![WhatsApp Image 2024-04-23 at 17 57 05_960a403f](https://github.com/prakhar1302/IR_Project/assets/142145465/d71f83b1-77b3-4e3b-bb1c-a73203b14ef9)

 ![WhatsApp Image 2024-04-23 at 17 56 16_adae2129](https://github.com/prakhar1302/IR_Project/assets/142145465/8ff0975f-7fcd-4c1d-a990-0447b1abd49f)

## Contributors
1)Akash kumar (MT23012)
akash23012@iiitd.ac.in

2)Prakhar Sharma (MT23060)
prakhar23012@iiitd.ac.in

3)Mo Rashid (MT23047)
rashid23047@iiitd.ac

4)Jafreen Rizvi (MT23040)
jafreen230@iiitd.ac.in

5)Harsh Choudhary (2020433)
harsh20433@iiitd.ac.in

6)Shazra Irshad (MT23089)
shazra23089@iiitd.ac.in
