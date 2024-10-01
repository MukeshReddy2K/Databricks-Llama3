# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC %pip install transformers
# MAGIC %pip install torch torchvision torchaudio

# COMMAND ----------

# MAGIC %restart_python or dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime


DATA_DIR = "/Workspace/Users/mukeshreddymavurapu@gmail.com/responses.xlsx"
df_data = pd.read_excel(DATA_DIR, sheet_name="Data")
df_questions = pd.read_excel(DATA_DIR, sheet_name="Questions")
df_data.head()

# COMMAND ----------

from huggingface_hub import login

# Log in to Hugging Face with your token
access_token = "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"
login(access_token)

# COMMAND ----------

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# COMMAND ----------

pip install --upgrade transformers


# COMMAND ----------

df_data['InstructorName'] = df_data['InstructorName'].fillna('Instructor, Dummy')

df_data["Processed"] = False
# we get the list of distinct sections (classNbr)
# then for each ClassNbr we filter out its responses
distinct_sections = df_data['classnbr'].unique()
print(f"Distinct sections: {len(distinct_sections)}")

df_summary = pd.DataFrame(
    columns=['term', 'Acadgroup', 'acadorg', 'classnbr', 'subject', 'catalog', 'dateInserted',
                'summary', 'qValue', 'Question', 'ModelName', 'ModelDescription'])


# COMMAND ----------

import ollama 

# ollama server code 
# def get_summary(model, comments, question):
#     prompt = "Summarize the comments asked for the following question " + question + ":"
#     prompt_comments = prompt + comments
#     response = ollama.chat(model=model, messages=[
#         {
#             'role': 'user',
#             'content': prompt_comments,
#         },
#     ])
#     return response['message']['content']

# COMMAND ----------

from datetime import datetime

for section_number in distinct_sections:
    if str(section_number) == "10343":
        df_section_responses = df_data.loc[df_data['classnbr'] == section_number]

        evalForm = int(df_section_responses["evalForm"].iloc[0])

        df_section_questions = df_questions.loc[df_questions['formnumber'] == evalForm]

        if df_section_responses['distEval'].iloc[0].any():
            df_section_questions = pd.concat([df_section_questions, df_questions[df_questions['formnumber'] == 23]],
                                                ignore_index=True)
        print(f"Section: {section_number}")
        subject = df_section_responses['subject'].iloc[0]
        df_section_responses = df_section_responses.copy()
        df_section_responses.loc[:, 'catalog'] = df_section_responses['catalog'].apply(lambda x: x[0] if isinstance(x, tuple) else x if pd.notna(x) else None )
        catalog = df_section_responses['catalog'].iloc[0],
        acadgroup = df_section_responses['acadgroup'].iloc[0]
        acadorg = df_section_responses['acadorg'].iloc[0]
        if evalForm is None:
            print("No form number found")
            
        else:
            for index, row in df_section_questions.iterrows():
                question = row['Question']
                qvalue = row["qValues"].lower()
                responses = df_section_responses[row["qValues"].lower()]
                responses = responses.dropna()
                responses = responses[responses != '']
                if len(responses) == 0:
                    summary = ''
                else:
                    responses = responses.tolist()
                    summary =''
                    responses = [response.replace('\n', ' ') for response in responses]
                    tokenizer.pad_token = tokenizer.eos_token

                    text = "Summarize the following comments in a third-person point of view, highlighting key points and insights with points: " + ' '.join(responses)

                    print("Model Start time:", datetime.now())
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

                    summary_ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=300,  # Higher max_length for flexibility
                    num_return_sequences=1,
                    do_sample=True,   # Allow sampling for more varied outputs
                    temperature=0.7,  # Adjust for variability
                    top_k=50,         # Use top-k sampling
                    top_p=0.9         # Use top-p sampling
                    )

                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summarized_text = summary

                    print("Generated Summary:", summarized_text)

                    print("Model End time:", datetime.now())
                    break
                    # description = 'LLaMA 3 (Large Language Model Meta AI) is a state-of-the-art language model designed ' \
                    #               'for natural language processing tasks. It is fine-tuned with 8 billion parameters. ' \
                    #               'trained on diverse datasets to follow complex instructions.'

                    # print(f"Question: {question}")

                    # print("Model start time:", datetime.now())

                    # text = get_summary('llama3', text, question)

                    # textList = ''

                    # print("Model end time:", datetime.now())

                    # if '/n/n' in text:
                    #     textList = text.split('/n/n', 1)
                    #     if len(textList) == 2:
                    #         summary = textList[1]
                    # else:
                    #     summary = text
                    # print(summary)
                    # model = 'LLaMA 3'
                    # df_summary.loc[len(df_summary)] = [term, acadgroup, acadorg, section_number, subject, catalog,
                    #                                    datetime.now(), summary, qvalue, question, model, description]


# COMMAND ----------

from ollama import OllamaClient

client = OllamaClient(server_url="http://localhost:5000")  # Replace with your server URL

# COMMAND ----------

# MAGIC %sh
# MAGIC ollama serve &

# COMMAND ----------

import ollama 

for section_number in distinct_sections:
    if str(section_number) == "10343":
        df_section_responses = df_data.loc[df_data['classnbr'] == section_number]

        evalForm = int(df_section_responses["evalForm"].iloc[0])

        df_section_questions = df_questions.loc[df_questions['formnumber'] == evalForm]

        if df_section_responses['distEval'].iloc[0].any():
            df_section_questions = pd.concat([df_section_questions, df_questions[df_questions['formnumber'] == 23]],
                                                ignore_index=True)
        print(f"Section: {section_number}")
        subject = df_section_responses['subject'].iloc[0]
        df_section_responses = df_section_responses.copy()
        df_section_responses.loc[:, 'catalog'] = df_section_responses['catalog'].apply(lambda x: x[0] if isinstance(x, tuple) else x if pd.notna(x) else None )
        catalog = df_section_responses['catalog'].iloc[0],
        acadgroup = df_section_responses['acadgroup'].iloc[0]
        acadorg = df_section_responses['acadorg'].iloc[0]
        if evalForm is None:
            print("No form number found")
            
        else:
            for index, row in df_section_questions.iterrows():
                question = row['Question']
                qvalue = row["qValues"].lower()
                responses = df_section_responses[row["qValues"].lower()]
                responses = responses.dropna()
                responses = responses[responses != '']
                if len(responses) == 0:
                    summary = ''
                else:
                    responses = responses.tolist()
                    summary =''

                    # remove newline from within each individual response/comment
                    responses = [response.replace('\n', ' ') for response in responses]

                    # concatenate the comments with a newline
                    text = ' '.join(responses)

                    prompt = "Summarize the comments asked for the following question " + question + ":"
                    prompt_comments = prompt + text
                    response = ollama.chat(model="llama3.1", messages=[
                        {
                            'role': 'user',
                            'content': prompt_comments,
                        },
                    ])
                    summary = response['message']['content']

                    print("Summarized text:", summary)
                    print("Model End time:", datetime.now())


# COMMAND ----------

import os
import subprocess

# Start the Ollama server in the background
# subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.Popen(["ollama", "serve", "--port", "11434"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# COMMAND ----------


