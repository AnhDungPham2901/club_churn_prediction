import pickle
import numpy as np
import gradio as gr
import pandas as pd
from lime import lime_tabular
import gpt_processing

def load_pretrained_model():
    with open('rf_model', 'rb') as model_file:
        loaded_rf_model = pickle.load(model_file)
    return loaded_rf_model

def get_feature_by_user_id(user_id):
    df = pd.read_excel('club_churn_train.xlsx')
    df = df[['ID']]
    df['ID'] = df['ID'].astype(int)
    X_test = pd.read_csv('X_test.csv')
    df = df.join(X_test, how="inner")
    feature_by_user = df[df['ID'] == user_id]
    feature_by_user.drop('ID', axis=1, inplace=True)
    return feature_by_user
    
def predict_and_explain(user_id: int):
    print(user_id)
    user_id = int(user_id)
    input_data = get_feature_by_user_id(user_id)
    if input_data.shape[0] == 0:
        return "NONE", "This users are not in churn list"
    X_train = pd.read_csv('X_train.csv')
    rf_model = load_pretrained_model()
    pred = rf_model.predict(input_data)
    # Get LIME explanation
    explainer = lime_tabular.LimeTabularExplainer(X_train.values, 
                                              feature_names=X_train.columns, 
                                              class_names=['INFORCE', 'CANCELLED'], 
                                              mode='classification')
    exp = explainer.explain_instance(input_data.values.ravel(), rf_model.predict_proba)
    explanation = exp.as_list()
    
    # Convert explanation to string
    explanation_str = '\n'.join([f"{item[0]}: {item[1]:.4f}" for item in explanation])
    
    # proceed explanation of LIME by using GPT
    beauti_explanation = gpt_processing.proceed_explanation(explanation_str)
    return f"{'CANCELLED' if pred[0] == 1 else 'INFORCE'}", beauti_explanation


interface = gr.Interface(fn=predict_and_explain,
                         inputs=["text"],
                         outputs=[
                             gr.outputs.Textbox(label="Prediction"),
                             gr.outputs.Textbox(label="Explanation")
                         ],
                         live=False)

interface.launch()
