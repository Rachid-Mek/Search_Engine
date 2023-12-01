import os
import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk
import nltk
from SearchEngine import SearchEngine

class UI(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Search Engine")

        # Initialize the customtkinter library
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('dark-blue')

        # Set the window size
        self.geometry('1000x600')

        # Create a frame to organize the elements
        self.frame = ctk.CTkFrame(master=self)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Interface element to enter a search term
        self.label = ctk.CTkLabel(master=self.frame, text="Termes:", font=("Roboto", 20))
        self.label.grid(row=0, column=0, pady=10, padx=20, sticky="e")

        self.entry = ctk.CTkEntry(master=self.frame, placeholder_text="Search for a term", font=("Roboto", 16), width=200)
        self.entry.grid(row=0, column=1, pady=10, padx=20, sticky="w")

        # Create checkbuttons for processing options
        self.token_label = ctk.CTkLabel(master=self.frame, text="Processing:", font=("Roboto", 20))
        self.token_label.grid(row=1, column=0, pady=10, padx=20, sticky="e")

        self.check_var_tokenization = ctk.StringVar(value="tokenization")
        self.check_tokenization = ctk.CTkCheckBox(master=self.frame, text="Tokenization", variable=self.check_var_tokenization, onvalue="tokenization", offvalue="off_token")
        self.check_tokenization.grid(row=1, column=1, pady=10, padx=20, sticky="w")

        self.check_var_porter = ctk.StringVar(value="on_porter")
        self.check_porter = ctk.CTkCheckBox(master=self.frame, text="PorterStem", variable=self.check_var_porter, onvalue="on_porter", offvalue="off_porter",width=150)
        self.check_porter.grid(row=1, column=2, pady=10, padx=20, sticky="w")

        # radio buttons for index type
        self.radio_label = ctk.CTkLabel(master=self.frame, text="Index type:", font=("Roboto", 20))
        self.radio_label.grid(row=2, column=0, pady=10, padx=20, sticky="e")

        self.var_index = ctk.StringVar(value="inverse")
        self.radio_inverse = ctk.CTkRadioButton(master=self.frame, text="Inverse", variable=self.var_index, value="inverse")
        self.radio_inverse.grid(row=2, column=1, pady=10, padx=20, sticky="w")

        self.radio_descriptive = ctk.CTkRadioButton(master=self.frame, text="Descriptive", variable=self.var_index, value="descriptive")
        self.radio_descriptive.grid(row=2, column=2, pady=10, padx=20, sticky="w")

        self.vector_model = ctk.CTkRadioButton(master=self.frame, text="Vector model", variable=self.var_index, value="vector")
        self.vector_model.grid(row=2, column=3, pady=10, padx=20, sticky="w")

        self.probabilistic_model = ctk.CTkRadioButton(master=self.frame, text="Probabilistic", variable=self.var_index, value="Probabilistic",width=150)
        self.probabilistic_model.grid(row=2, column=4, pady=10, padx=20, sticky="w")

        # create two input fields one for K and the other for B
        self.k_label = ctk.CTkLabel(master=self.frame, text="K:", font=("Roboto", 20))
        self.k_label.grid(row=3, column=4, pady=10, padx=20, sticky="e")

        self.k_entry = ctk.CTkEntry(master=self.frame, placeholder_text="K", font=("Roboto", 16),width=50)
        self.k_entry.grid(row=3, column=5, pady=10, padx=20, sticky="w")

        self.b_label = ctk.CTkLabel(master=self.frame, text="B:", font=("Roboto", 20))
        self.b_label.grid(row=3, column=6, pady=10, padx=20, sticky="e")

        self.b_entry = ctk.CTkEntry(master=self.frame, placeholder_text="B", font=("Roboto", 16),width=50)
        self.b_entry.grid(row=3, column=7, pady=10, padx=20, sticky="w")

        # Create the OptionMenu widget with initial options
        self.selected_model_type = ctk.StringVar(value="Produit Scalaire")
        self.vector_model_type = ctk.CTkOptionMenu(master=self.frame, values=["Produit Scalaire", "Cosine Measure", "Indice de Jaccard"], width=150)
        self.vector_model_type.grid(row=3, column=3, pady=10, padx=20, sticky="w")

        self.file_path = None
        self.folder_path = None

        self.button_choose_file = ctk.CTkButton(master=self.frame, text="Select a file", font=("Roboto", 16), command=self.choose_file)
        self.button_choose_file.grid(row=1, column=0, pady=10, padx=20, sticky="e")

        self.button_choose_folder = ctk.CTkButton(master=self.frame, text="Select a folder", font=("Roboto", 16), command=self.choose_folder)
        self.button_choose_folder.grid(row=2, column=0, pady=10, padx=20, sticky="e")

        self.button_search = ctk.CTkButton(master=self.frame, text="Search", font=("Roboto", 20), command=self.search)
        self.button_search.grid(row=3, column=1, pady=10, padx=20, sticky="w")

        # Create the Treeview widget with three columns
        self.tree = ttk.Treeview(master=self.frame, columns=('', '', '', ''), style="Custom.Treeview")
        self.tree.grid(row=4, column=0, pady=10, padx=20, sticky="nsew", columnspan=8)

        # Set the minimum width for each column
        min_width = 150
        self.tree.column('#0', minwidth=min_width, width=min_width, stretch=tk.NO)
        self.tree.column('#1', minwidth=min_width, width=min_width, stretch=tk.NO)
        self.tree.column('#2', minwidth=min_width, width=min_width, stretch=tk.NO)
        self.tree.column('#3', minwidth=min_width, width=min_width, stretch=tk.NO)

        # Configure the grid to expand the Treeview widget vertically
        self.frame.grid_rowconfigure(5, weight=1)
        for i in range(8):
            self.frame.grid_columnconfigure(i, weight=1)

        # Create a custom style for the column headings
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Roboto", 16))

        # Apply the custom style to the column headings
        self.tree.tag_configure("Treeview.Heading", font=("Roboto", 16))

        # Make the table fill the full height of the window
        self.frame.grid_rowconfigure(3, weight=1)
        self.search_engine = SearchEngine()

    
    
    def choose_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if self.file_path:
            print("Fichier choisi:", self.file_path)

    def choose_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            print("Dossier choisi:", self.folder_path)

    def update_columns(self):
        selected_index = self.var_index.get()
        columns = ()
        column_headers = ()

        if selected_index == 'inverse':
            columns = ("Terme", "Nom du fichier", "Fréquence", "Weight")
            column_headers = ("Terme", "Nom du fichier", "Fréquence", "Weight")
        elif selected_index == 'descriptive':
            columns = ("Nom du fichier", "Terme", "Fréquence", "Weight")
            column_headers = ("Nom du fichier", "Terme", "Fréquence", "Weight")
        else:
            columns = ("doc_id", "Relevance")
            column_headers = ("doc_id", "Relevance")

        # Remove any existing columns and headings
        for column in self.tree.get_children():
            self.tree.delete(column)
            self.tree["columns"] = ()

        self.tree["columns"] = columns
        for column in columns:
            self.tree.column(column, width=200, anchor="w")
            self.tree.heading(column, text=column_headers[columns.index(column)])

    def search(self):
        term = self.entry.get()
        method = 'Porter' if self.check_var_porter.get() == 'on_porter' else 'Lancaster'
        split = False if self.check_var_tokenization.get() == 'tokenization' else True

        results = []

        if len(term) == 1:
            term = nltk.PorterStemmer().stem(term)
            index = 'Inverse' if self.var_index.get() == 'inverse' else 'Descripteurs'
            if not term:
                if self.folder_path:
                    file_name = f"{index}{method}{'Split' if split else 'Token'}.txt"
                    results = self.search_engine.get_data(file_name)
                elif self.file_path:
                    results = self.search_engine.display_all_terms(self.file_path, method, split)
                    print(results)
            else:
                if self.file_path:
                    freq = self.search_engine.Normalizer(self.file_path, method, split)
                    if term in freq:
                        results.append((term, os.path.basename(self.file_path), freq[term]))
                    else:
                        results.append((term, os.path.basename(self.file_path), 0))
                elif self.folder_path:
                    results = self.search_engine.search_term(term, method, split)
                else:
                    results = self.search_engine.search_term(term, method, split)
                    print(results)
        else:
            if self.var_index.get() == 'vector':
                results = self.search_engine.RSV(term, self.vector_model_type.get())
            elif self.var_index.get() == 'Probabilistic':
                k = float(self.k_entry.get())
                b = float(self.b_entry.get())
                results = self.search_engine.BM25(term, b, k)

        self.update_columns()

        # Clear the existing data in the Treeview
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Create a custom style for the Treeview cell text
        style = ttk.Style()
        style.configure("Custom.Treeview.Cell", font=("Roboto", 12))
        # Apply the custom style to the cell text
        self.tree.tag_configure("Custom.Treeview.Cell", font=("Roboto", 12))

        # Set padding between rows
        style.configure("Custom.Treeview", padding=(10, 5))

        # Align text to the left
        self.tree.tag_configure("Custom.Treeview.Cell", anchor="w")

        # Update the Treeview with the new results
        if results:
            for result in results:
                if self.var_index.get() == 'inverse':
                    self.tree.insert("", "end", values=result, tags=("Custom.Treeview.Cell",))
                elif self.var_index.get() == 'descriptive':
                    self.tree.insert("", "end", values=(result[1], result[0], result[2], result[3]), tags=("Custom.Treeview.Cell",))
                else:
                    self.tree.insert("", "end", values=(result[0], result[1]), tags=("Custom.Treeview.Cell",))
        else:
            self.tree.insert("", "end", values=("No results found", "", "", ""), tags=("Custom.Treeview.Cell",))

if __name__ == "__main__":
    ui = UI()
    ui.mainloop()