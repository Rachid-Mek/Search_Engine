import os
import re
import math
import nltk

class SearchEngine:
    def get_data(self, file_name):
        results = []
        file_path = os.path.join('output', file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stored_term, doc_id, frequency, weight = line.split()
                results.append((stored_term, doc_id, frequency, weight))
        return results

    def display_all_terms(self, file_path, method, split_option):
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
                    term_results.append((term, os.path.basename(file_path), frequency, weight))

                # Append results for the current term to the overall results list
                results.extend(term_results)

        return results

    def sort_column(self, tv, col, reverse):
        tv.delete(*tv.get_children())
        items = sorted(tv.get_children(''), key=lambda x: tv.set(x, col), reverse=reverse)
        for item in items:
            tv.insert('', 'end', values=[tv.item(item, 'values')])
        # Toggle the sort order
        global sort_order
        sort_order = not reverse

    def search_term(self, term, method, split_option):
        results = []
        # Getting the data from all files that contain the term
        N = len(os.listdir('Collection'))  # Total number of documents in the collection
        for doc_id, file_name in enumerate(os.listdir('Collection'), start=1):
            file_path = os.path.join('Collection', file_name)
            freq = self.Normalizer(file_path, method, split_option)
            if term in freq:
                # calculating the number of documents containing the term in freq
                ni = [term in self.Normalizer(os.path.join('Collection', file_name), method, split_option) for file_name in os.listdir('Collection')].count(True)
                frequency = freq[term] # Frequency of the term in the current document
                max_freq = max(freq.values()) # Maximum frequency of any term in the current document
                weight = self.calculate_weight(N, ni, frequency, max_freq) 
                results.append((term, doc_id, frequency, weight))

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

    def RSV(self, query, modele):
        word = query.split()
        word = [nltk.PorterStemmer().stem(word) for word in word]
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
            return sorted_dict

        elif(modele == 'Indice de Jaccard'):
            jac =[vect[i] / ((q[i] + v[i]) - vect[i]) for i in range (len(vect))]
            result_dict = {i + 1: jac[i] for i in range(len(jac)) if jac[i] != 0}
            sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted_dict

        elif(modele == 'Produit Scalaire'):
            #sorting the vec by the weight and keeping the doc_id and removing the weight = 0
            dict = {}
            for i in range(len(vect)):
                if vect[i] != 0:
                    dict[i+1] = vect[i]
            dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
            return dict

    def BM25(self, query, B=1.5, K=2):
        words = query.split()
        words = [nltk.PorterStemmer().stem(word) for word in words]
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

    def logic_query(self, query):
        '''cette fonction va verifier si la requete est valide ou pas'''
        pattern = r"^(not\s+)?\w+(\s+(and|or)\s+(not\s+)?\w+)*$"
        return bool(re.match(pattern, query, re.IGNORECASE))

# Example usage:
if __name__ == '__main__':
    search_engine = SearchEngine()
    print(search_engine.logic_query('terme and terme or '))