from tkinter import font as tkFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import K_medoid as kmd
from tkinter import messagebox
from tkinter import StringVar

class MovieApp:

    def __init__(self, master, dataframe, clusters, edit_rows=[]):
        self.root = master
        self.root.minsize(width=600,height=400)
        self.root.title('IMDb Movie Predictor')

        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.lab_opt = {'background': 'darkgreen', 'foreground': 'white'}

        #the dataframe
        self.df = dataframe
        self.clusters = clusters
        self.dat_cols = list(self.df)

        if edit_rows:
            self.dat_rows = edit_rows
        else:
            self.dat_rows = range(len(self.df))
        self.rowmap = {i: row for i, row in enumerate(self.dat_rows)}

        #subset the data and convert the list of rows for viewing
        self.sub_data = self.df.loc[self.dat_rows, self.dat_cols]
        self.sub_datastring = self.sub_data.to_string(index = False, col_space = 13).split('\n')
        self.title_string = self.sub_datastring[0]
        self.results = ""
        self.clicked = 0

        #save the format of the lines, so it can be updated without re-running
        self._get_line_format(self.title_string)

        #fill in the main frame
        self._fill()

        self.update_history = []
        self.movie_t = StringVar()
        self.no_user_rev = StringVar()
        self.buget = StringVar()
        self.no_critic_reviews = StringVar()
        self.fb_likes = StringVar()
        self.usr_votes = StringVar()
        self.duration = StringVar()

        self.tree = None

    #widgets

    def _fill(self):
        self.canvas = tk.Canvas(self.main)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        self._init_scroll()
        self._init_lb()
        self._pack_config_scroll()
        self._pack_bind_lb()
        self._fill_listbox()
        self._make_editor_frame()

    #scroll bars
    def _init_scroll(self):
        self.scrollbar = tk.Scrollbar(self.canvas, orient = "vertical")
        self.xscrollbar = tk.Scrollbar(self.canvas, orient ='horizontal')

    def _pack_config_scroll(self):
        self.scrollbar.config(command=self.lb.yview)
        self.xscrollbar.config(command=self._xview)
        self.scrollbar.pack(side="right", fill="y")
        self.xscrollbar.pack(side="bottom", fill="x")

    def _onMouseWheel(self, event):
        self.title_lb.yview("scroll", event.delta, "units")
        self.lb.yview("scroll", event.delta, "units")
        return "break"

    def _xview(self, *args):
        self.lb.xview(*args)
        self.title_lb.xview(*args)


    #main listbox
    def _init_lb(self):
        self.title_lb = tk.Listbox(self.canvas, height=1,
                                   font=tkFont.Font(self.canvas,
                                                    family="Courier",
                                                    size=14),
                                   yscrollcommand=self.scrollbar.set,
                                   xscrollcommand=self.xscrollbar.set,
                                   exportselection=False)

        self.lb = tk.Listbox(self.canvas,
                             font=tkFont.Font(self.canvas,
                                              family="Courier",
                                              size=14),
                             yscrollcommand=self.scrollbar.set,
                             xscrollcommand=self.xscrollbar.set,
                             exportselection=False,
                             selectmode=tk.EXTENDED)

    def _pack_bind_lb(self):
        self.title_lb.pack(fill=tk.X)
        self.lb.pack(fill="both", expand=True)
        self.title_lb.bind("<MouseWheel>", self._onMouseWheel)
        self.lb.bind("<MouseWheel>", self._onMouseWheel)

    def _fill_listbox(self):

        #fill the listbox with rows from the dataframe
        self.title_lb.insert(tk.END, self.title_string)
        for line in self.sub_datastring[1:]:
            self.lb.insert(tk.END, line)
            self.lb.bind('<ButtonRelease-1>', self._listbox_callback)
        self.lb.select_set(0)


    def _listbox_callback(self):

        #when a list box item in selected
        items = self.lb.curselection()
        if items:
            new_item = items[-1]
            datVal = str(self.df.ix[self.rowmap[new_item], self.opt_var.get()])
            self.entry_box_old.config(state=tk.NORMAL)
            self.entry_box_old.delete(0, tk.END)
            self.entry_box_old.insert(0, datVal)
            self.entry_box_old.config(state=tk.DISABLED)

    #frame for editing
    def _make_editor_frame(self):

        #make a frame for editing dataframe rows
        self.editorFrame = tk.Frame(self.main, bd=2, padx=2, pady=2, relief=tk.GROOVE)
        self.editorFrame.pack(fill=tk.BOTH, side=tk.LEFT)

        #column editor
        self.col_sel_lab = tk.Label(self.editorFrame, text='Show Clusters:', **self.lab_opt)
        self.col_sel_lab.grid(row=0, columnspan=2, sticky=tk.W + tk.E)

        self.show_cluster = tk.Button(self.editorFrame, text='View Clusters', command=self.plot_graph)
        self.show_cluster.grid(row=0, column=3, columnspan=2, sticky=tk.W + tk.E)

        self.col_sel_lab = tk.Label(self.editorFrame, text='Process Clusters:', **self.lab_opt)
        self.col_sel_lab.grid(row=1, columnspan=2, sticky=tk.W + tk.E)

        self.show_cluster = tk.Button(self.editorFrame, text='Process', command=self.process_data_set)
        self.show_cluster.grid(row=1, column=3, columnspan=2, sticky=tk.W + tk.E)


    #updating lines
    def _rewrite(self):

        #re-writing the dataframe string in the listbox
        new_co_vals = self.df.ix[self.row, self.data_cols].astype(str).tolist()
        new_line = self._make_line(new_co_vals)

        if self.lb.cget('state') == tk.DISABLED:
            self.lb.config(state=tk.NORMAL)
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)
            self.lb.config(state=tk.DISABLED)
        else:
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)


    def _get_line_format(self, line):

        #save the format of the title string, store positions of the columns breaks
        pos = [1+ line.find(' ' + n) + len(n) for n in self.dat_cols]
        self.entry_length = [pos[0]] + \
                            [p2 - p1 for p1, p2 in zip(pos[:-1], pos[1:])]

    def _make_line(self, col_entries):

        #add a new line to the database in the correct format
        new_line_entries = [('{0: >%d}' % self.entry_length[i]).format(entry)
                            for i, entry in enumerate(col_entries)]
        new_line = "".join(new_line_entries)

        return new_line

    def plot_graph(self):
        markers = ['bo', 'go', 'ro', 'c+', 'm+', 'y+']
        clusters = self.clusters

        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                plt.plot(df[0], df[1], markers[i])

        plt.xlabel('IMDb Scores')
        plt.ylabel('Gross')
        plt.title('K-Medoid Clusters')
        plt.legend()
        plt.show()

    def assign_target(self, row):
        x = row['movie_title']
        clusters = self.clusters

        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                if df[2] == x:
                    row['cluster'] = 'cluster' + str(i)

        return row

    def show_results(self):
        if self.clicked >=1:
            messagebox.showinfo("Results of the decision tree model", self.results)
        else:
            messagebox.showerror("Please process the clusters","Please press the process button to generate the results")


    def makeform(self):

        if self.clicked ==0:
            messagebox.showerror("Please process the clusters","Please press the process button before this")
        else:
            master = tk.Toplevel(self.root)
            master.geometry("350x350")

            tk.Label(master, text="Movie Title").grid(row=0, columnspan=10)
            tk.Label(master, text="Number of user reviews").grid(row=1, columnspan=10)
            tk.Label(master, text="Budget").grid(row=2, columnspan=10)
            tk.Label(master, text="Number of critic reviews").grid(row=3, columnspan=10)
            tk.Label(master, text="Movie facebook likes").grid(row=4, columnspan=10)
            tk.Label(master, text="Number of user votes").grid(row=5, columnspan=10)
            tk.Label(master, text="Duration").grid(row=6, columnspan=10)

            e1 = tk.Entry(master, textvariable=self.movie_t)
            e2 = tk.Entry(master, textvariable=self.no_usr_rev)
            e3 = tk.Entry(master, textvariable=self.budget)
            e4 = tk.Entry(master, textvariable=self.no_critic_reviews)
            e5 = tk.Entry(master, textvariable=self.fb_likes)
            e6 = tk.Entry(master, textvariable=self.usr_votes)
            e7 = tk.Entry(master, textvariable=self.duration)

            e1.grid(row=0, column=11)
            e2.grid(row=1, column=11)
            e3.grid(row=2, column=11)
            e4.grid(row=3, column=11)
            e5.grid(row=4, column=11)
            e6.grid(row=5, column=11)
            e7.grid(row=6, column=11)

            btn = tk.Button(master, text='Predict', command=self.process_form)
            btn.grid(row=8, column=3, columnspan=5, sticky=tk.W + tk.E)


    def process_form(self):

        a = self.no_usr_rev.get()
        b = self.budget.get()
        c = self.no_critic_reviews.get()
        d = self.fb_likes.get()
        e = self.usr_votes.get()
        f = self.duration.get()

        if a == "" or b == "" or c == "" or d == "" or e == "" or f == "":
            messagebox.showerror("Null Values", "Please fill in the empty spaces")
        df = pd.DataFrame([[int(a), float(b), int(c), int(d), int(e), int(f)]],
                          columns=['num_user_for_reviews', 'budget', 'num_critic_for_reviews', 'movie_facebook_likes',
                                   'num_voted_users', 'duration'])

        result = 'The movie ' + self.movie_t.get() + ' is a part of \'' + self.tree.predict(df)
        result += '\'. Refer to the classification report for more details'
        messagebox.showinfo("Prediction of the movie", result)


    def process_data_set(self):

        # choosing features for decision tree
        columns = ['movie_title', 'num_user_for_reviews', 'budget'
            , 'num_critic_for_reviews', 'movie_facebook_likes', 'num_voted_users', 'duration']

        df = self.df[columns]
        df = df.apply(self.assign_target, axis=1)
        df.drop(labels=['movie_title'], axis=1, inplace=True)



if __name__ == '__main__':
    columns = ['movie_title', 'num_user_for_reviews', 'budget', 'num_critic_for_reviews', 'movie_facebook_likes',
               'num_voted_users', 'duration', 'gross', 'imdb_score']

    # loading dataset
    df = pd.read_csv('movie_metadata.csv').dropna(axis=0).reset_index(drop=True)

    dataset = df[['gross', 'imdb_score', 'movie_title']]
    dataset = dataset.values.tolist()

    clusters = kmd.K_medoid(dataset, 5, np.inf, 0)

    root = tk.Tk()
    editor = MovieApp(root, df[columns], clusters)

    root.mainloop()


