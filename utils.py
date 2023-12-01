import math
import os
import nltk

def search_term(term, method, split_option):
    results = []
    # Getting the data from all files that contain the term
    N = len(os.listdir('Collection'))  # Total number of documents in the collection
    for doc_id, file_name in enumerate(os.listdir('Collection'), start=1):
        file_path = os.path.join('Collection', file_name)
        freq = Normalizer(file_path, method, split_option)
        if term in freq:
            # calculating the number of documents containing the term in freq
            ni = [term in Normalizer(os.path.join('Collection', file_name), method, split_option) for file_name in os.listdir('Collection')].count(True)
            frequency = freq[term] # Frequency of the term in the current document
            max_freq = max(freq.values()) # Maximum frequency of any term in the current document
            weight = calculate_weight(N, ni, frequency, max_freq) 
            print(N, ni, frequency, max_freq, weight)
            results.append((term, doc_id, frequency, weight))

    return results

def Normalizer(input_file, method, split =False):
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

def calculate_weight(N, ni, freq, max_freq):
    return ((freq / max_freq) * math.log10((N / ni) + 1))

def create_files(collection_directory, output_directory, output_prefix):
    methods = ['Porter', 'Lancaster']
    split_options = [False, True]
    
    for method in methods:
        for split_option in split_options:
            term_doc_mapping = {}  # Dictionary to store term to document mapping
            doc_term_count = {}  # Dictionary to count the number of documents containing each term

            for doc_id, filename in enumerate(os.listdir(collection_directory), start=1):
                input_file = os.path.join(collection_directory, filename)
                term_frequency = Normalizer(input_file, method, split_option)
                max_freq = max(term_frequency.values())
                N = len(os.listdir(collection_directory))
                
                for term, frequency in term_frequency.items():
                    if term not in doc_term_count:
                        doc_term_count[term] = 1  # Initialize count for the current term
                    else:
                        doc_term_count[term] += 1  # Increment the count for the current document

                    ni = [term in Normalizer(os.path.join('Collection', file_name), method, split_option) for file_name in os.listdir('Collection')].count(True)
                    weight = calculate_weight(N, ni, frequency, max_freq)

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

# Example usage:
collection_directory = 'Collection'
output_directory = 'output'
os.makedirs(output_directory, exist_ok=True)
create_files(collection_directory, output_directory, "Descripteurs")
create_files(collection_directory, output_directory, "Inverse")