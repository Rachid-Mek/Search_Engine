import gradio as gr
from SearchEngine import SearchEngine
from prerocessing import extract_query_information

#----------------------------- Search Engine -----------------------------#
search_engine = SearchEngine()
def search(search_term, use_query_dataset, query_id, tokenization, lancaceter, display_option, pertinence, Vector_space_model, K, B):

    if use_query_dataset:
        if query_id is not None:
            query_id = int(query_id)
            search_term = search_engine.get_query(query_id)

    term = search_term
    method = 'Lancester' if lancaceter else 'Porter' 
    tokenize = 'Token' if tokenization else 'Split'
    collection = 'Descripteurs' if display_option == 'Descriptive' else 'Inverse'

    if term != '':
        results = search_engine.dataframe(inverse=collection, lancaseter= lancaceter, tokenize=tokenization)
        #check if term is result id
        # if term in [str(x) for x in results['doc_id']]:
        #     term = int(term)  # Convert term to an integer
        #     results = results[results['doc_id'] == term]
        #     #printing somme des frequences
        #     print(results['frequency'].sum())
        if not pertinence:
            term = search_engine.process_query(term, method=lancaceter, split=tokenization)
            term = term[0]
            results = results[results['term'] == term]

        if pertinence:
            term = search_engine.process_query(term, method=lancaceter, split=tokenization)
            results = search_engine.RSV(term, Vector_space_model, K, B,method, tokenize,collection)
      
            if use_query_dataset:
                query_id = int(query_id)
                metrics_df = search_engine.calculate_metrics(query_id, results)
                plt_file = search_engine.plot(query_id, results)    
                table_style = 'width: 100%; max-width: 800px; height: 50%; max-height:400px; overflow: auto;'
                return (
                    f'<div style="{table_style}">{results.reset_index().to_html(index=False)}</div>' +
                    f'<hr/>' +
                    f'<div style="{table_style}">{metrics_df.to_html(index=False)}</div>' +
                    f'<hr/>',
                    plt_file if plt_file else 'temp/RI.jpg'  
                )

    else:
        results = 'Please enter a query'

    if results is None:
        return 'No results found', 'temp/RI.jpg'  

    if type(results) == str:
        return results + f'<hz/>', 'temp/RI.jpg'  
    
    table_style = 'width: 100%; max-width: 800px; height: 50%; max-height:500px; overflow-y: auto;'

    return f'<div style="{table_style}">{results.reset_index().to_html(index=False)}</div> ' + f'<hz/>' , 'temp/RI.jpg'  


#----------------------------- Graphical User Interface -----------------------------#
dataset = extract_query_information('output_lisa/Judgement.txt')
query_numbers = dataset['query_num']
query_numbers = [None] + [str(x) for x in query_numbers]


interface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Search Terms", value=""),
        gr.Checkbox(label="Query Dataset", value=False),
        gr.Dropdown(label="Query", choices=query_numbers),
        gr.Checkbox(label="Tokenization", value=True),
        gr.Checkbox(label="Lancaster"),
        gr.Radio(choices=["Inverse", "Descriptive"], label="Search Method", value="Descriptive"),
        gr.Checkbox(label="Pertinence"),
        gr.Dropdown(label="Model", choices=["Produit Scalaire", "Cosine Measure", "Indice de Jaccard", "Probabilistic", "Bool"]),
        gr.Number(label="K", value=2.0),
        gr.Number(label="B", value=0.5),
    ],
    outputs=[gr.HTML(), gr.Image()],
    allow_flagging="never",
    title="Search Engine",
    theme=gr.themes.Soft(),
)   

interface.launch(inbrowser=True)