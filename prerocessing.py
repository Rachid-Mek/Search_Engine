from collections import Counter
import os
import pandas as pd
import nltk

def load_processed_docs(folder_path):
    """Reads all files in a given Folder

    Returns:
        dict:dictionnary of all files that have been read
    """

    docs_content = dict()

    # Ensure the path is a directory
    if os.path.isdir(folder_path):
        # List all files in the directory
        files = os.listdir(folder_path)

        for file_name in files:
            # Check if the file is a text file
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                content = pd.read_csv(file_path, sep=" ")

                # Store the text content in the dictionary
                docs_content[file_name.replace(".txt", "")] = content

    return docs_content


def load_descripteurs_and_inverse():
    """returns a dictionnary with all descripteurs and inverse documents
     example : {"inverse_split_port":datadrame(term,doc,frequency,weight),...}
    """

    descripteurs= load_processed_docs("output_lisa")

    return descripteurs

def extract_query_information(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    query_info = {'query_num': [], 'relevant': [], 'docs': []}
    current_query = None
    relevant_refs_line = False
    docs = False

    for line in lines:
        if line.startswith('Query'):
            current_query = int(line.split()[1])
            relevant_refs_line = True
        elif relevant_refs_line:
            relevant_refs = [int(ref) for ref in line.split() if ref.isdigit()]
            query_info['query_num'].append(current_query)
            query_info['relevant'].append(relevant_refs[0])
            relevant_refs_line = False
            docs = True
        elif docs:
            docs = [int(doc) for doc in line.split() if doc != '-1']
            query_info['docs'].append(docs)
            docs = False

    # Convert to dataframe
    query_info = pd.DataFrame(query_info)

    return query_info

def create_document_dictionary(document_data):
    
    """create a dict where key is doc_id and value is the content of doc

    Returns:
        dict:{"1":str representing the content of doc1 , etc..}
        """
    
    documents = document_data.split("********************************************\n")
    document_dict = {}

    for document in documents:
        lines = document.split('\n')
        if lines:
            document_id = lines[0].split()[-1]
            content = ' '.join(lines[1:]).strip() 
            document_dict[document_id] = content

    return document_dict

def read_documents(file_path):
    with open(file_path, 'r') as file:
        document_data = file.read()
    return document_data

def split_tokenizer(txt):
    return [token for token in txt.split()]

def regex_tokenizer(txt,regex='(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*'):
    reg = nltk.RegexpTokenizer(regex)
    return reg.tokenize(txt)

def stop_remove(tokens):
    stop = nltk.corpus.stopwords.words('english')
    return [term.lower() for term in tokens if term.lower() not in stop]

def porter_stem(tokens):
    porter = nltk.PorterStemmer()
    return[porter.stem(term) for term in tokens]

def lancester_stem(tokens):
    # lancester = nltk.PorterStemmer()
    lancester = nltk.LancasterStemmer()
    return[lancester.stem(term) for term in tokens]


def files_preprocessor(documents):
    
    """This function applies tokenization ,stop words removal and stemming
    it takes as input documents which is a dictionnary with values as documents content

    Returns:
        _type_: _description_
    """
    
    #{file_name:dict of frequencies,file_name:dict of frequencies}
    all_docs_frequencies_split_port=dict()
    all_docs_frequencies_split_lancaster=dict()
    all_docs_frequencies_regex_port=dict()
    all_docs_frequencies_regex_lancaster=dict()
    
    for id , content in documents.items():
        
        #get tokens
        splitted_tokens = split_tokenizer(content)
        regex_tokens = regex_tokenizer(content,'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
        #remove stopwords
        splitted_tokens = stop_remove(splitted_tokens)
        regex_tokens = stop_remove(regex_tokens)
        
        #stemming
        splitted_tokens_porter = porter_stem(splitted_tokens)
        splitted_tokens_lancester = lancester_stem(splitted_tokens)
        
        regex_tokens_porter = porter_stem(regex_tokens)
        regex_tokens_lancester = lancester_stem(regex_tokens)
        
        #frequencies
        
        all_docs_frequencies_split_port[id] = dict(Counter(splitted_tokens_porter))
        all_docs_frequencies_split_lancaster[id] = dict(Counter(splitted_tokens_lancester))
        all_docs_frequencies_regex_port[id] = dict(Counter(regex_tokens_porter))
        all_docs_frequencies_regex_lancaster[id] = dict(Counter(regex_tokens_lancester))
        

    descripteur = dict()
    descripteur["PorterSplit"] = all_docs_frequencies_split_port
    descripteur["LancesterSplit"] = all_docs_frequencies_split_lancaster
    descripteur["PorterToken"] = all_docs_frequencies_regex_port
    descripteur["LancesterToken"] = all_docs_frequencies_regex_lancaster
    
    return descripteur


def unique_occurences(descripteurs):
    
    all_terms_occ = dict()
    
    for descripteur , docs_freq_dict in descripteurs.items():
        terms_occ = dict()
        for doc_id , doc_freq in docs_freq_dict.items():
            
            for term , freq in doc_freq.items():          
                terms_occ[term] = terms_occ.get(term,0) + 1 
        
        all_terms_occ[descripteur] = terms_occ  
    
    return all_terms_occ          