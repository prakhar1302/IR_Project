import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#Function to build
def getLLamaresponse(input_text):
    ### LLama2 model
    llm=CTransformers(model='Model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Write advantage and disdvantage of {input_text} medicine
        within 100 words.
            """
    
    prompt=PromptTemplate(input_variables=["input_text"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(input_text=input_text))
    print(response)
    return response




st.set_page_config(page_title = "Medicine Details",
page_icon = ' 🤖',
layout = 'centered',
initial_sidebar_state = 'collapsed')

st.header("Medicine ")

input_text=st.text_input("Enter Medicine name for details")

submit = st.button("Generate")

##Final

if submit:
    st.write(getLLamaresponse(input_text))





