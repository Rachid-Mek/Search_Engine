import os
import re
import math
import nltk
import pandas as pd

class SearchEngine:

    def split_tokenizer(self , txt):
        return [token for token in txt.split()]

    def regex_tokenizer(self,txt,regex='(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*'):
        reg = nltk.RegexpTokenizer(regex)
        return reg.tokenize(txt)

    def stop_remove(self,tokens):
        stop = nltk.corpus.stopwords.words('english')
        return [term.lower() for term in tokens if term.lower() not in stop]

    def porter_stem(self, tokens):
        porter = nltk.PorterStemmer()
        return[porter.stem(term) for term in tokens]

    def lancester_stem(self, tokens):
        # lancester = nltk.PorterStemmer()
        lancester = nltk.LancasterStemmer()
        return[lancester.stem(term) for term in tokens]

    def display_all_terms(self, file_path, method, split_option , inverse = False):
        results = []
        if file_path:
            doc_term_count = {}  # Dictionary to count the number of documents containing each term
            term_frequency = self.Normalizer(file_path, method, split_option)
            max_freq = max(term_frequency.values())
            N = len(os.listdir('Collection'))

            for term, frequency in term_frequency.items():
                term_results = []  # List to store results for the current term
                for doc_id, filename in enumerate(os.listdir('Collection'), start=1):
                    if term not in doc_term_count:
                        doc_term_count[term] = 1  # Initialize count for the current term
                    else:
                        doc_term_count[term] += 1  # Increment the count for the current document

                    ni = doc_term_count[term]
                    weight = self.calculate_weight(N, ni, frequency, max_freq)
                    if inverse:
                        term_results.append((term, doc_id, frequency, weight))
                    else:
                        term_results.append((doc_id, term, frequency, weight))

                # Append results for the current term to the overall results list
                results.extend(term_results)

        results_df = pd.DataFrame(results, columns=['term', 'doc_id', 'frequency', 'weight']) if inverse else pd.DataFrame(results, columns=['doc_id', 'term', 'frequency', 'weight'])
        return results_df

    def search_term(self,query, lancaster, tokenize, inverse):
        query = self.process_query(query, lancaster, tokenize)
        results = []
        # Getting the data from all files that contain the term
        results = self.dataframe(tokenize ,lancaster, inverse)
        #query is a list of terms
        for term in query:
            results = results[results['term'] == term]
        return results

    def Normalizer(self, input_file, method, split =False):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        if split:
            Termes = text.split()
        else:
            # Tokenization
            ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
            Termes = ExpReg.tokenize(text)
        MotsVides = nltk.corpus.stopwords.words('english')
        TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
        if method == 'Porter':
            Porter = nltk.PorterStemmer()
            TermesNormalisation = [Porter.stem(terme) for terme in TermesSansMotsVides]
        elif method == 'Lancaster':
            Lancaster = nltk.LancasterStemmer()
            TermesNormalisation = [Lancaster.stem(terme) for terme in TermesSansMotsVides]

        TermesFrequency = {}

        for terme in TermesNormalisation:
            if (terme in TermesFrequency.keys()):
                TermesFrequency[terme] += 1
            else:
                TermesFrequency[terme] = 1

        return TermesFrequency

    def calculate_weight(self, N, ni, freq, max_freq):
        return ((freq / max_freq) * math.log10((N / ni) + 1))

    def create_files(self, collection_directory, output_directory, output_prefix):
        methods = ['Porter', 'Lancaster']
        split_options = [False, True]
        
        for method in methods:
            for split_option in split_options:
                term_doc_mapping = {}  # Dictionary to store term to document mapping
                doc_term_count = {}  # Dictionary to count the number of documents containing each term

                for doc_id, filename in enumerate(os.listdir(collection_directory), start=1):
                    input_file = os.path.join(collection_directory, filename)
                    term_frequency = self.Normalizer(input_file, method, split_option)
                    max_freq = max(term_frequency.values())
                    N = len(os.listdir(collection_directory))
                    
                    for term, frequency in term_frequency.items():
                        if term not in doc_term_count:
                            doc_term_count[term] = 1  # Initialize count for the current term
                        else:
                            doc_term_count[term] += 1  # Increment the count for the current document

                        ni = [term in self.Normalizer(os.path.join('Collection', file_name), method, split_option) for file_name in os.listdir('Collection')].count(True)
                        weight = self.calculate_weight(N, ni, frequency, max_freq)

                        if output_prefix == "Descripteurs":
                            doc_term_key = f"{doc_id} {term}"
                        else:
                            doc_term_key = f"{term} {doc_id}"

                        term_doc_mapping[doc_term_key] = (frequency, weight)

                output_filename = f"{output_prefix}{method}{'Split' if split_option else 'Token'}.txt"
                output_file = os.path.join(output_directory, output_filename)

                with open(output_file, 'w', encoding='utf-8') as f:
                    # Sort the items (key and (frequency, weight)) and iterate over them
                    sorted_items = sorted(
                        term_doc_mapping.items(), key=lambda item: item[0]
                    )
                    for key, (frequency, weight) in sorted_items:
                        if output_prefix == "Descripteurs":
                            doc_id, term = key.split()
                            f.write(f"{doc_id} {term} {frequency} {weight}\n")
                        else:
                            term, doc_id = key.split()
                            f.write(f"{term} {doc_id} {frequency} {weight}\n")

    def BM25(self, query,K=2, B=1.5):
        words = self.process_query(query, 'Porter', False)
        dl, term_freq, ni = [0.0] * len(os.listdir('Collection')), [[0.0] * len(os.listdir('Collection')) for _ in range(len(words))], [0.0] * len(words)
        NUMBER_OF_DOC = len(os.listdir('Collection'))
        with open('output/DescripteursPorterToken.txt','r',encoding='utf-8') as f:
            for line in f :
                doc_id , term , frequency , weight = line.split()
                dl[int(doc_id) - 1] += int(frequency)
                for i , word in enumerate(words) :
                    if term == word :
                        term_freq[i][int(doc_id) - 1] += int(frequency)
                        ni[i] = sum(1 for file_name in os.listdir('Collection') if word in self.Normalizer(os.path.join('Collection', file_name), 'Porter', False))

        avdl = (sum(dl) / NUMBER_OF_DOC)
        BM25_scores = [0.0] * len(os.listdir('Collection'))
        for i in range(NUMBER_OF_DOC):
            somme = 0.0 
            for j in range(len(words)):
                somme += (term_freq[j][i]/(K * ((1- B) + B * (dl[i]/avdl)) + term_freq[j][i]) * (math.log10((NUMBER_OF_DOC - ni[j] + 0.5)/(ni[j] + 0.5))))
            BM25_scores[i] = somme

        result_dict = {i + 1: BM25_scores[i] for i in range(len(BM25_scores)) if BM25_scores[i] != 0}
        sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        print(words)
        print(f'dl :{dl}')
        print(f'avdl : {avdl}')
        print(f'term frequency : {term_freq}')
        print(f'ni : {ni}')
        print(f'BM25 : {sorted_dict}')

        return sorted_dict

    def is_valid_query(self, query):
        '''cette fonction va verifier si la requete est valide ou pas'''
        pattern = r"^(not\s+)?\w+(\s+(and|or)\s+(not\s+)?\w+)*$"
        return bool(re.match(pattern, query, re.IGNORECASE))

    #def Bool_model(self, query):
        if not self.is_valid_query(query):
            return 'Not a valid query'

        # Initialize the result dictionary with zeros
        result_dict = {i + 1: 0 for i in range(len(os.listdir('Collection')))}

        query = query.lower()
        tokens = [nltk.PorterStemmer().stem(token) for token in query.split()]

        # Read data from CSV file
        df = pd.read_csv('output/DescripteursPorterToken.txt', sep=" ")
        # Group by the first index of the column because we don't have a name for it
        df_grouped = df.groupby(df.columns[0])[df.columns[1]].apply(list).reset_index(name='term')
        # Rename the first column to 'doc_id'
        df_grouped.rename(columns={df_grouped.columns[0]: 'doc_id'}, inplace=True)

        not_tokens = set()
        for i, token in enumerate(tokens):
            if token == 'not':
                not_tokens.add(tokens[i + 1])

        # Check if each document satisfies the query
        for i, doc_terms in enumerate(df_grouped['term']):
            # Use 'or' to return 1 if at least one token is present in the document
            # Use 'and' to return 1 if all tokens are present in the document
            if ('or' in tokens and any((token in doc_terms) for token in tokens)) or \
            ('and' in tokens and all((token in doc_terms) for token in tokens)):
                result_dict[i + 1] = 1
            else: # check the individual tokens
                if tokens[0] == 'not':
                    if tokens[1] in doc_terms:
                        result_dict[i + 1] = 0
                    else:
                        result_dict[i + 1] = 1
                else:
                    if tokens[0] in doc_terms:
                        result_dict[i + 1] = 1
                    else:
                        result_dict[i + 1] = 0

        # return the result dictionary as a list not sorted
        return result_dict.items()
            
    def Bool_model(self, query):
        if not self.is_valid_query(query):
            return 'Not a valid query'

        # Initialize the result dictionary with zeros
        result_dict = {i + 1: 0 for i in range(len(os.listdir('Collection')))}

        query = query.lower()
        tokens = [nltk.PorterStemmer().stem(token) for token in query.split()]

        # Read data from CSV file
        df = pd.read_csv('output/DescripteursPorterToken.txt', sep=" ")
        # Group by the first index of the column because we don't have a name for it
        df_grouped = df.groupby(df.columns[0])[df.columns[1]].apply(list).reset_index(name='term')
        # Rename the first column to 'doc_id'
        df_grouped.rename(columns={df_grouped.columns[0]: 'doc_id'}, inplace=True)

        # using postfix stack to evaluate the query and save the result in result_dict
        postfix_stack = []

        def evaluate_expression(operand1, operator, operand2):
            if operator == 'and':
                return operand1 and operand2
            elif operator == 'or':
                return operand1 or operand2

        for token in tokens:
            if token not in {'and', 'or', 'not'}:
                # Check if the token is a valid document ID
                if token in result_dict:
                    postfix_stack.append(result_dict[token])
                else:
                    print(f"Invalid document ID: {token}")
                    return

            elif token == 'not':
                operand = postfix_stack.pop()
                postfix_stack.append(not operand)
            else:  # 'and' or 'or'
                operand2 = postfix_stack.pop()
                operand1 = postfix_stack.pop()
                result = evaluate_expression(operand1, token, operand2)
                postfix_stack.append(result)

        # return the result dictionary as a list sorted
        sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_dict

    def RSV(self, query, modele,k,b):
        word = self.process_query(query, 'Porter', False)
        q , v , vect = [0.0]*len(os.listdir('Collection')) , [0.0]*len(os.listdir('Collection')) , [0.0]*len(os.listdir('Collection'))
        with open('output/DescripteursPorterToken.txt', 'r', encoding='utf-8') as f:
            for line in f:
                doc_id, term, frequency, weight = line.split()
                q[int(doc_id)-1] += float(weight) ** 2  # q
                if term in word:
                    vect[int(doc_id)-1] += float(weight)  # scalar product vector
                    v[int(doc_id)-1] += 1  # v

        if(modele == 'Cosine Measure'):
            norm_query = [math.sqrt(q[i]) for i in range(len(q))]  # calculating the sqrt for each element in q
            norm_doc = [math.sqrt(v[i]) for i in range(len(v))]  # calculating the sqrt for each element in v
            # calculating the cosine similarity with a check for zero division
            cos = [vect[i] / (norm_query[i] * norm_doc[i]) if norm_query[i] != 0 and norm_doc[i] != 0 else 0 for i in range(len(vect))]
            result_dict = {i + 1: cos[i] for i in range(len(cos)) if cos[i] != 0}
            sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

        elif(modele == 'Indice de Jaccard'):
            jac =[vect[i] / ((q[i] + v[i]) - vect[i]) for i in range (len(vect))]
            result_dict = {i + 1: jac[i] for i in range(len(jac)) if jac[i] != 0}
            sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

        elif(modele == 'Produit Scalaire'):
            #sorting the vec by the weight and keeping the doc_id and removing the weight = 0
            dict = {}
            for i in range(len(vect)):
                if vect[i] != 0:
                    dict[i+1] = vect[i]
            sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        
        elif(modele == 'Probabilistic'):
            sorted_dict = self.BM25(query, k, b)
        
        elif(modele == 'bool'):
            sorted_dict = self.Bool_model(query)
        
        # saving the result in a dataframe
        result_df = pd.DataFrame(sorted_dict, columns=['doc_id', 'Relevance'])
        return result_df

    def process_query(self , query,method,split=False):
        
        tokens = self.regex_tokenizer(query) if not split else self.split_tokenizer(query)
        tokens = self.stop_remove(tokens)
        tokens = self.porter_stem(tokens) if method == "Porter" else self.lancester_stem(tokens)
    
        return tokens
    
    def dataframe(self, tokenize= True, lancaseter = False, inverse = 'Descripteurs'):
        
        """inverse file
        
        query : the searched sentence
        tokenize : regex or split
        stemming : port for porter else lancester
        
        Returns:
            data frame on inverse or decsriptor file from output folder
        """  
        file_name = f"{inverse}{'Lancaster' if lancaseter else 'Porter'}{'Token' if tokenize else 'Split'}.txt"
        file_path = os.path.join('output', file_name)
        df = pd.read_csv(file_path, sep=" ")
        # naming the columns
        df.columns = ['term', 'doc_id', 'frequency', 'weight'] if inverse == 'Inverse' else ['doc_id', 'term', 'frequency', 'weight']
        return df

# Example usage:
if __name__ == '__main__':
    search_engine = SearchEngine()
    query = '1'
    resutls = search_engine.dataframe(inverse= 'Descripteurs')
    # displaying the query results in a dataframe where term = query
    print(resutls[resutls['doc_id'] == int(query)])