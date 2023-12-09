import os
import gradio as gr
import nltk
from SearchEngine import SearchEngine

#----------------------------- Search Engine -----------------------------#
search_engine = SearchEngine()

def search(search_term, tokenization, lancaceter, display_option, pertinence ,Vector_space_model,K,B):
    term = search_term
    method = 'Lancaster' if lancaceter else 'Porter'
    split = tokenization
    collection = 'Descripteurs' if display_option == 'descriptive' else 'Inverse'

    results = []

    if term != '':
        results = search_engine.dataframe(inverse=collection)
        print(term)
        if term in ['1', '2', '3', '4', '5', '6']:
            print(f"Filtering by doc_id: {term}")
            term = int(term)  # Convert term to an integer
            results = results[results['doc_id'] == term]
        else:
            results = results[results['term'] == term]

        if pertinence:
            print(f"Using RSV for pertinence: {Vector_space_model},{K},{B}")
            results = search_engine.RSV(term, Vector_space_model, K, B)
            print(f"RSV Results:\n{results}")
        elif results.empty:
            print(f"Using regular search for pertinence")
            results = search_engine.search_term(term, method, split, collection)
            print(f"Regular Search Results:\n{results}")
    else:
        results = 'Please enter a query'
    

    if results.empty:
        return 'No results found'

    if type(results) == str:
        return results

    return results.reset_index().to_html(index=False)


#----------------------------- Graphical User Interface -----------------------------#
iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Search Term"),
        gr.Checkbox(label="Tokenization", value=True),
        gr.Checkbox(label="lancaceter"),
        gr.Radio(choices= ["inverse", "descriptive"], label="Search Method", value="descriptive"),
        gr.Checkbox(label="pertinence"),
        gr.Dropdown(label="Model", choices= ["Produit Scalaire", "Cosine Measure", "Indice de Jaccard", "Probabilistic", "Bool"]),
        gr.Number(label="K",value=2.0),
        gr.Number(label="B",value=1.5),
    ],
    outputs=gr.HTML(),
    live=True,
    allow_flagging="never",
    title="Search Engine",
    theme=gr.themes.Soft()
)

iface.launch(inbrowser=True)
