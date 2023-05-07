### LIBRARIES ###
import boto3
import re
import time
import os
import openai
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gradio as gr
import asyncio
import aiohttp
import pandas as pd
import requests
import json
import re
import pdfkit
import boto3
import json
import sys
import time
import pandas as pd
from user_agent import generate_user_agent
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pdfminer.high_level import extract_text
from tqdm import tqdm
tqdm.pandas()


### CONSTANTS ###
openai.api_key = os.getenv("OPENAI_API_KEY")
mymodel = '/Users/slidarey/Downloads/rbc-demo/90000'
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
dataset_name = 'data copy.parquet'
original_dataset = pd.read_parquet('data.parquet')

### FUNCTIONS ###
def open_file(filepath):
    '''
    Description: Used to open the 'prompt_answer.txt' file.
    Input: filepath -> path to the file being read.
    '''
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def content_embedding(content, model_name):
    '''
    generating embedding for model
    '''
    #Load the model
    # model = SentenceTransformer('sentence-transformers/' + model_name)
    model = SentenceTransformer(model_name)

    embeddings = model.encode(content)
    return embeddings

def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    '''
    Desciption: Calls the OpenAI to answer the users question.
    '''
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()

            text = re.sub('\s+', ' ', text) # replacing all whitespace with a single space
            return text

        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            time.sleep(1)

def answering_question(question, ticker, year):
    # debugging
    print('Answering Question')
    dataset = pd.read_parquet(dataset_name)
    
    embedding = content_embedding(question, model_name=mymodel)
    filtered_dataset = dataset.copy().reset_index(drop=True)

    try:
        year = int(year)
    except ValueError as e:
        year = 0
        print(e)

    if (ticker != "None") and (year != 0):
        filtered_dataset = dataset[(dataset['ticker'] == ticker) & (dataset['year_of_filing'] == year)].reset_index(drop=True)
    elif ticker != "None":
        filtered_dataset = dataset[(dataset['ticker'] == ticker)].reset_index(drop=True)
    elif year != 0:
        filtered_dataset = dataset[(dataset['year_of_filing'] == year)].reset_index(drop=True)
    else:
        print(filtered_dataset.shape)
        pass
    
    if filtered_dataset.shape[0] == 0:
        return 'Please get the filing first. Filing does not exist.'

    # filtered_dataset = dataset.copy().reset_index(drop=True)
    out = filtered_dataset['embedding'].apply(lambda x: util.cos_sim(x, embedding)).sort_values(ascending=False)
    bi_df = filtered_dataset.iloc[out.head(10).index].copy()
    out = bi_df['context'].apply(lambda x: cross_encoder.predict([question, x])).sort_values(ascending=False)
    result = (out.head(5).index)
    
    print(out.head())

    if out.head(1).values[0] < 5:
        return 'Unable to find answer to your question. Try a different query.'

    passage_list = list()

    for i, index in enumerate(result):
        context = filtered_dataset.iloc[index]['context']
        if i != 0:
            context = context.split('\n', 1)[1]
        passage_list.append(context)

    
    passage = ' '.join(passage_list)
    prompt = (open_file('/Users/slidarey/Downloads/rbc-demo/prompt_answer.txt')
                .replace('<<PASSAGE>>', passage).replace('<<QUERY>>', question))
    print(prompt)
    
    answer = gpt3_completion(prompt)
    # answer = 'Question was answered.'
    return answer

# splitting the concatenated_output into four sentences and put in pandas DataFrame
def split_into_sentences(content, no_sntcs=4):
    '''
    This function splits the text into sentences and then groups them into desired number of sentences.
    :param content: text to be split
    :param no_sntcs: number of sentences to be grouped
    :param split_overlap: if True, then the sentences will be split with overlap
    :param no_of_tokens: number of tokens to be grouped
    :param overlap: number of sentences to be overlapped
    :return: list of dictionaries with grouped sentences
    '''
    sentences = re.split('\.\s([A-Z][\w\d]+)', content)
    clnd_sntc = list()
    grpd_sntc = list()

    # splitting the whole text into sentences
    for i in range(len(sentences)):
        if i == 0:
            # first sentence
            clnd_sntc.append(sentences[i].strip() + '.')
        elif (i+2 == len(sentences)):

            # for last sentence = appending captured split pattern to it's sentence
            clnd_sntc.append((sentences[i] + sentences[i+1]).strip())
        elif ((i % 2) != 0):

            # for middle sentences - appending captured split pattern to it's sentence
            clnd_sntc.append(sentences[i] + sentences[i+1] + '.')
        else:
            pass

    # check if number of sentences is more than the desired number of sentences for grouping
    if len(clnd_sntc) >= no_sntcs:

        # to know how many group of sentences can be formed
        for i in range(len(clnd_sntc)//no_sntcs):

            # group and append desired set of sentences
            header = ' '.join(clnd_sntc[(i*no_sntcs):((i+1)*no_sntcs)])
            grpd_sntc.append(header)

            # check if remaining length is less than the desired number of group of sentences
            if (((len(clnd_sntc) - ((i+1)*no_sntcs)) < no_sntcs) and ((len(clnd_sntc) - ((i+1) * no_sntcs)) != 0)):

                # append remaining sentences
                header = ' '.join(clnd_sntc[((i+1)*no_sntcs):])
                grpd_sntc.append(header)
            else:
                pass
    else:
        # return all sentence as 1 group because total number of sentence is not up to desired number of group of sentences
        header = ' '.join(clnd_sntc)
        grpd_sntc.append(header)

    return grpd_sntc

def create_df(list_of_texts):
    '''
    Function to create dataframe from the series of list of dictionaries.
    '''
    master_list = list(list_of_texts)

    df = pd.DataFrame(master_list, columns=['texts'])
    return df

def get_filing(filing_ticker, filing_year):
    # debugging
    print('Getting your File')
    init_data = pd.read_parquet(dataset_name)

    # Generate a user agent string for Chrome
    user_agent = generate_user_agent(navigator='chrome')

    lookup_headers = {
        "referer": "https://sec-api.io",
        "user-agent": user_agent
    }

    filing_ticker = str(filing_ticker)
    filing_year = int(filing_year)
    file_link = str()
    company_name = str()
    company_ticker = str()
    filing_id = str()
    data = {
            "query": {"query_string": {
                                        "query": "formType:\"10-K\" AND ticker:(" + filing_ticker + ") AND filedAt:[" + str(
                                        filing_year) + "-01-01 TO " + str(filing_year+1) + "-01-01] AND documentFormatFiles.type:*21*"
                                        }
                    }, "sort": [{
                                    "filedAt": {"order": "desc"}
                                    }]}

    try:
        response = requests.post(
            "https://api.sec-api.io/", headers=lookup_headers, json=data, timeout=10)
    except requests.exceptions.Timeout:
        print('The request timed out')

    if response.status_code == 200:
        parsed_json = json.loads(response.text)
        hits = parsed_json['total']['value']
        if hits > 0:
            company_name = parsed_json['filings'][0]['companyName']
            company_ticker = parsed_json['filings'][0]['ticker']
            examine_link = parsed_json['filings'][0]['linkToFilingDetails']
            filing_id = parsed_json['filings'][0]['id']
            if examine_link.find('ix?doc=/') != -1:
                file_link = examine_link.replace('ix?doc=/', '')
            else:
                file_link = examine_link
        else:
            print('No hits')
            return """ <p style="text-align: left; font-size: 12px; color: gray; font-style: italic;"> The filing you're looking for doesn't exist yet. </p> """
    else:
        print(response.status_code)


    # check if filing already exist in the database using filing_id
    if init_data['filing_id'].isin([filing_id]).any():
        return """ <p style="text-align: left; font-size: 12px; color: gray; font-style: italic;"> This filing already exists. </p> """

    # getting the html content of the 10-K filing and removing any image content and table content

    html_to_pdf_retries = 0
    get_page_headers = {"user-agent": user_agent}

    try:
        response = requests.get(file_link, headers=get_page_headers, timeout=10)
    except requests.exceptions.Timeout:
        print('The request timed out')

    if response.status_code == 200:
        html = response.text
        html_without_images = re.sub(r'<img.*?>', '', html)
        html_without_tables = re.sub(r'<table.*?>.*?</table>', '', html_without_images)
        print(response.status_code)
    else:
        print(response.status_code)

    #####################################################
    # changed html_without_tables to html_without_images
    pdfkit.from_string(html_without_images, 'output.pdf')

    PDF_read = extract_text('output.pdf')
    e_PDF_read = PDF_read.replace('\n\n', ' ').replace('\n', ' ')
    e_PDF_read = re.sub(r'\s+', ' ', e_PDF_read)

    # grouped_sentences = split_into_sentences(concatenated_output)
    grouped_sentences = split_into_sentences(e_PDF_read)
    df_texts = create_df(grouped_sentences)

    def create_context(content):
        '''
        Function to create context from the dataset.
        '''
        context = 'Company: ' + company_name + \
            ' (' + company_ticker + ')' + '\n' + content['texts']
        return context

    # creating context using the company_name and new_text column
    df_texts['context'] = df_texts.apply(create_context, axis=1)

    # creating context using the company_name and new_text column
    df_texts['embedding'] = df_texts['context'].progress_apply(content_embedding, model_name=mymodel)
    df_texts['filing_id'] = filing_id
    df_texts['company_name'] = company_name
    df_texts['ticker'] = company_ticker
    df_texts['year_of_filing'] = filing_year
    df_texts.drop(columns=['texts'], inplace=True)

    # merge the new dataframe with the existing dataframe
    merge_df = pd.concat([init_data, df_texts], ignore_index=True)
    merge_df.to_parquet(dataset_name, compression='gzip')

    return """ <p style="text-align: left; font-size: 12px; color: gray; font-style: italic;"> Filing parsed. </p> """


# UI CREATION
with gr.Blocks(title="RBC 10-K filing Q&A - Demo", theme=gr.themes.Soft()) as demo:
    
    title = gr.Markdown(""" <p style="text-align: center; font-size: 24px; font-weight: bold;"> RBC 10-K filing Q&A - Demo </p> """)
    with gr.Row():
        with gr.Column(scale=1):
            get_filing_title = gr.Markdown(""" <p style="text-align: center; font-size: 16px;"> Get company 10-K filing</p> """)
            filing_ticker = gr.Textbox(label="Ticker", interactive=True, placeholder="TSLA")
            filing_year = gr.Textbox(label="Year", interactive=True, placeholder="2016")
            get_filing_done = gr.Markdown()
            get_btn = gr.Button("GET", interactive=True)
            get_btn.click(fn=get_filing, inputs=[filing_ticker, filing_year], outputs=get_filing_done, api_name="get")
        with gr.Column(scale=4):
            with gr.Row():
                ticker_choices = sorted(list(original_dataset['ticker'].unique()))
                ticker_choices.insert(0, "None")
                year_choices = sorted(list(original_dataset['year_of_filing'].unique()))
                year_choices.insert(0, 'None')
                ticker = gr.Dropdown(label="Ticker", choices=ticker_choices, interactive=True, value="None")
                year = gr.Dropdown(label="Year", choices=year_choices, interactive=True, value="None")

            note = gr.Markdown("""<p style="font-size: 12px; color: gray; font-style: italic;"> Please note that the year corresponds to the year the filing was filed.</p>""")

            question = gr.Textbox(label="Question", interactive=True, placeholder="Ask a question")
            answer = gr.Textbox(label="Answer")
            ask_btn = gr.Button("ASK", interactive=True)
            ask_btn.click(fn=answering_question, inputs=[question, ticker, year], outputs=answer, api_name="ask")


# with gr.Blocks(title="RBC 10-K filing Q&A - Demo", theme=gr.themes.Soft()) as demo:
#     title = gr.Markdown(""" <p style="text-align: center; font-size: 24px; font-weight: bold;"> RBC 10-K filing Q&A - Demo </p> """)
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             get_filing_title = gr.Markdown(""" <p style="text-align: center; font-size: 16px;"> Get company 10-K filing</p> """)
#             filing_ticker = gr.Textbox(label="Ticker", interactive=True, placeholder="TSLA")
#             filing_year = gr.Textbox(label="Year", interactive=True, placeholder="2016")
#             get_filing_done = gr.Markdown()
#             get_btn = gr.Button("GET", interactive=True)
#             get_btn.click(fn=get_filing, inputs=[filing_ticker, filing_year], outputs=get_filing_done, api_name="get")
        
#         with gr.Column(scale=4):
#             with gr.Row():
#                 ticker = gr.Dropdown(label="Ticker", choices=sorted(list(original_dataset['ticker'].unique())), interactive=True, value="None")
#                 year = gr.Dropdown(label="Year", choices=sorted(list(original_dataset['year_of_filing'].unique())), interactive=True, value="None")

#             note = gr.Markdown("""<p style="font-size: 12px; color: gray; font-style: italic;"> Please note that the year corresponds to the year the filing was filed.</p>""")

#             question = gr.Textbox(label="Question", interactive=True, value="What was google net income in 2016?")
#             answer = gr.Textbox(label="Answer")
#             ask_btn = gr.Button("ASK", interactive=True)
#             ask_btn.click(fn=answering_question, inputs=[question, ticker, year], outputs=answer, api_name="ask")


demo.launch(share=True)


# DATASET
# apple, amazon, microsoft, netflix = 2010 - 2022
# meta/facebook = 2013 - 2022
# google = 2016 - 2022
