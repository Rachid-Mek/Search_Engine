import os
import re
import math
import tempfile
import mpld3
import nltk
from nltk.stem import PorterStemmer , LancasterStemmer
import pandas as pd
import matplotlib.pyplot as plt

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
        return sorted_dict

    def infix_to_postfix(self,expression):
        operators = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        stack = []

        def apply_operator(operator):
            while stack and operators.get(stack[-1], 0) >= operators[operator]:
                output.append(stack.pop())
            stack.append(operator)

        for token in expression.split():
            if token in operators:
                apply_operator(token)
            else:
                output.append(token)

        while stack:
            output.append(stack.pop())

        return ' '.join(output)

    def get_bool_value(self, term,doc,inverse_dict,stemmer):
        term=stemmer.stem(term)

        v=inverse_dict.get((term,doc),0)
        if v==0:
            return False
        else:
            return True
    
    def evaluate_postfix_expression(self, expression, doc,inverse,stemmer):
        stack = []

        def apply_operator(operator):
            if operator == 'NOT':
                operand = stack.pop()
                result = not bool(operand)
            else:
                operand2 = stack.pop()
                operand1 = stack.pop()
                if operator == 'AND':
                    result = bool(operand1) and bool(operand2)
                elif operator == 'OR':
                    result = bool(operand1) or bool(operand2)

            stack.append(result)

        for token in expression.split():
            if token  not in {'NOT', 'AND', 'OR'}:
                stack.append(self.get_bool_value(token,doc,inverse,stemmer))
            elif token in {'NOT', 'AND', 'OR'}:
                apply_operator(token)
            else:
                stack.append(bool(token))

        return stack[0]
    
    def search_bool(self,term,stemmer):

        if stemmer=="porter":
            stemmer = PorterStemmer()
        else:
            stemmer = LancasterStemmer()

        df=self.dataframe(inverse="Inverse")
        result_dict={}
        for idx,row in df.iterrows():
            result_dict[(row["term"],row["doc_id"])]=row['frequency']

        final_dict={}
        for i in range(6):
            postfix=self.infix_to_postfix(term)
            val=self.evaluate_postfix_expression(postfix,i+1,result_dict,stemmer)

            if val :
                final_dict[i+1]= 'Yes'
            else:
                final_dict[i+1]= 'no'

        sorted_dict = list(final_dict.items())
        # filtring the result
        sorted_dict = [x for x in sorted_dict if x[1] == 'Yes']
        return sorted_dict

    def RSV(self, query, modele,k,b):
        word = self.process_query(query, False, False)
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
        
        elif(modele == 'Bool'):
            sorted_dict = self.search_bool(query, 'porter')
        
        # saving the result in a dataframe
        result_df = pd.DataFrame(sorted_dict, columns=['doc_id', 'Relevance'])
        return result_df

    def process_query(self , query, method, split=False):
        
        tokens = self.regex_tokenizer(query) if not split else self.split_tokenizer(query)
        tokens = self.stop_remove(tokens)
        tokens = self.porter_stem(tokens) if not method else self.lancester_stem(tokens)
    
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

    def pertitant_query(self, query, method, split = False):
        """pertinant query
        
        query : the searched sentence
        tokenize : regex or split
        stemming : port for porter else lancester
        
        Returns:
            data frame on inverse or decsriptor file from output folder
        """  
        tokens = self.regex_tokenizer(query) if not split else self.split_tokenizer(query)
        tokens = self.stop_remove(tokens)
        tokens = self.porter_stem(tokens) if method == "Porter" else self.lancester_stem(tokens)
        return tokens

    def get_query(self, query_num):
        """Get the query from the Queries.txt file. 
            exemple if query_num = 1 then return the first query which is in the first line of the file
        """
        with open('output/Queries.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == int(query_num) - 1:
                    return line.strip()
   
    # --------------------------------- metrics --------------------------------- #
    def calculate_metrics(self, query_num, results):
        """Calculate precision, recall, f-measure, precision@5, and precision@10 for a given query."""
        df_jugement = pd.read_csv('output/Judgement.txt', sep="\t", names=['query_num', 'doc_id'])
        df_jugement = df_jugement[df_jugement['query_num'] == query_num]
        df_jugement = df_jugement['doc_id'].tolist()
        df_results = results['doc_id'].tolist()

        # Calculate the number of pertinent documents
        pertinant = len(set(df_jugement).intersection(df_results))

        # Calculate metrics
        precision = pertinant / len(df_results)
        recall = pertinant / len(df_jugement)
        f_measure = 2 * precision * recall / (precision + recall)
        doc_pertinent = list(set(df_jugement).intersection(df_results))
        doc_pertinent_5 = len(set(df_results[:5]).intersection(doc_pertinent))
        doc_pertinent_10 = len(set(df_results[:10]).intersection(doc_pertinent))

        precision_5 = doc_pertinent_5 / 5 
        precision_10 = doc_pertinent_10 / 10 

        # Create a DataFrame with the metrics
        metrics_df = pd.DataFrame({
            'ID': [query_num],
            'Precision': [precision],
            'Recall': [recall],
            'F_Measure': [round(f_measure, 4)],
            'Precision@5': [precision_5],
            'Precision@10': [precision_10]
        })

        return metrics_df

    def get_recal_prec_per_doc(self, query_num, results):

        df_jugement = pd.read_csv('output/Judgement.txt', sep="\t", names=['query_num', 'doc_id'])
        df_jugement = df_jugement[df_jugement['query_num'] == query_num]
        df_jugement = df_jugement['doc_id'].tolist()
        df_results = results['doc_id'].tolist()
        print(df_results)
        doc_pertinent = list(set(df_jugement).intersection(df_results))
        print(doc_pertinent)

        precisions = []
        recalls = []
        pertinant = 0
        for i in range(1, 11):
            if i <= len(df_results) and df_results[i-1] in doc_pertinent:
                pertinant = pertinant + 1
                precisions.append(round((pertinant / i), 4))
                recalls.append(round((pertinant / len(df_jugement)), 4))
            else:
                precisions.append(round((pertinant / i), 4))
                recalls.append(round((pertinant / len(df_jugement)), 4))

        return precisions, recalls
    
    def interpolate(self, query_num, results):
        precisions, recalls = self.get_recal_prec_per_doc(query_num, results)
        print(f'precisons :{precisions}')
        print(f'recalls :{recalls}')
        interpolated_precisions = []
        recall = [i/10 for i in range(11)]
        i = 0
        j = 0
        while i < len(precisions):
            while j < len(recall):
                if recalls[i] >= recall[j]:
                    
                    j += 1
                    # return the max precision
                    interpolated_precisions.append(max(precisions[i:]))
                else:
                    i += 1
                    break

            if j == len(recall) or i == len(recalls):
                if (len(interpolated_precisions) != 11):
                    x = len(interpolated_precisions)
                    for i in range(x, 11):
                        interpolated_precisions.append(0)
                break

        return interpolated_precisions, recall
    
    def plot(self, query_num, results):
        interpollated, rec = self.interpolate(query_num, results)
        print(f'length of interpolated : {interpollated}')
        print(f'recall : {rec}')
        # Create Matplotlib figure
        fig, ax = plt.subplots()
        ax.plot(rec, interpollated, 'o-', color='red', label='Interpolated Precision')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall curve')
        ax.legend()


        # Specify a different temporary directory
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=temp_dir) as temp_file:
            plt.savefig(temp_file.name, format='png')
            temp_file_path = temp_file.name

        # Close the Matplotlib figure
        plt.close(fig)

        return temp_file_path

         

# Example usage:
if __name__ == '__main__':
    query_num = 1 
    results = pd.DataFrame({'doc_id': [2, 5, 3, 6, 4]})
    search_engine = SearchEngine()

    term = search_engine.plot(query_num, results)
    print(term)