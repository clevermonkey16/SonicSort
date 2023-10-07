#test
#Todo: make plot transparent, load bars, download button

import os
import sys
import shutil
import librosa
import pandas as pd
import numpy
import numpy as np
import csv
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import matplotlib as mat_agg
import matplotlib.pyplot as plt
import os.path
mat_agg.use('Agg')
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pygame
import glob

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QCompleter,
    QPushButton,
    QGroupBox,
    QWidget,
    QComboBox,
    QScrollArea
)
from PyQt6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt



suffix = ".wav"
#metadata = pd.DataFrame()
extracted_features = []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        pygame.mixer.init()
        # Set the background image
        pixmap = QPixmap("background.jpg")
        background_label = QLabel(self)
        pixmap_resized = pixmap.scaled(920, 600)
        background_label.setPixmap(pixmap_resized)

        background_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.centralwidget = background_label
        self.setCentralWidget(self.centralwidget)

        lay = QHBoxLayout(self.centralwidget)

        pushButton1 = QLabel(" ", self.centralwidget)
        spaces = ""
        for i in range(160):
            spaces += " "
        self.pushButton2 = QLabel(spaces, self.centralwidget)

        lay.addWidget(pushButton1)
        lay.addWidget(self.pushButton2)

        self.leftLayout = QVBoxLayout(pushButton1)


        pixmap2 = QPixmap("sonicsort.jpg")
        logo = QLabel(pushButton1)
        pixmap_resized2 = pixmap2.scaledToWidth(300)
        logo.setPixmap(pixmap_resized2)
        self.leftLayout.addWidget(logo)

        self.rightLayout = QVBoxLayout(self.pushButton2)
        button = QPushButton("Select Folder", self)
        button.clicked.connect(self.get_folder)
        self.leftLayout.addWidget(button)

        # combo box
        self.combobox = QComboBox()
        self.combobox.setEditable(True)
        self.soundlist = []
        self.soundlist.append(" ")
        #for filename in os.listdir("filtered_samples"):
        #    self.soundlist.append(filename)
        self.combobox.addItems(self.soundlist)
        self.combobox.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combobox.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
        completer = QCompleter(self.soundlist, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.combobox.setCompleter(completer)

        self.leftLayout.addWidget(self.combobox)
        self.combobox.currentIndexChanged.connect(self.on_combobox_changed)
        #######

        # scroll
        scroll = QScrollArea()
        widget = QWidget()
        self.vbox = QVBoxLayout()
        self.obj_list = []
        for i in range(1, 50):
            obj = QPushButton("Sound " + str(i), self)
            text = obj.text()
            obj.clicked.connect(lambda ch, a=i: self.on_button_click(a))
            self.obj_list.append(obj)
            self.vbox.addWidget(obj)
        widget.setLayout(self.vbox)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.leftLayout.addWidget(scroll)
        ######

        fig.savefig('figure.png', transparent=True)

        pixmapPlot = QPixmap("figure.png")
        self.plot = QLabel(self.pushButton2)
        pixmapPlot_resized = pixmapPlot.scaledToWidth(600)
        self.plot.setPixmap(pixmapPlot_resized)
        self.rightLayout.addWidget(self.plot)


    def on_combobox_changed(self):
        name = self.combobox.currentText()
        print(name)
        self.open_file(name)
        # do your code

    def on_button_click(self, a):
        # should play the sound, get file path
        self.play(a)
        print(a)

    def get_folder(self):
        global folderpath
        folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.getDatabase(folderpath)
        global extracted_features
        extracted_features = []
        self.create_extracted_features_arr()
        self.run_file_model()
        self.reload_combobox()
        print("done")

    def getDatabase(self, folder):
        file_path = folder
        if os.path.exists(file_path):
            print("Path loaded")
        else:
            print("Path not found")

        if not os.path.exists(os.path.join(os.getcwd(), "filtered_samples")):
            os.mkdir(os.path.join(os.getcwd(), "filtered_samples"))
        else:
            shutil.rmtree((os.path.join(os.getcwd(), "filtered_samples")))
            os.mkdir(os.path.join(os.getcwd(), "filtered_samples"))

        if os.path.exists(file_path):
            for root, dirs, files in os.walk(file_path):
                for d in dirs:
                    if d == "Z. Presets":
                        dirs.remove(d)
                    if d == "Y. MIDI":
                        dirs.remove(d)
                # print(dirs)

                for f in files:
                    # print(f)
                    try:
                        if f.endswith(suffix):
                            # print(f)
                            length = librosa.get_duration(filename=os.path.join(root, f))
                            if length < 1:
                                # librosa.load(os.path.join(root, f))
                                print(f, length)
                                shutil.copy2(os.path.join(root, f), "filtered_samples")
                    except Exception:
                        print("Exception: ", f)
                        pass
        with open('directory.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File name"])

            for root, dirs, files in os.walk('filtered_samples'):
                files.sort()
                for f in files:
                    print(f)
                    if f.endswith(suffix):
                        writer.writerow([f])

        global folder_loaded
        folder_loaded = True

    def features_extractor(self,file):
        try:
            # samplingFrequency, signalData = wavfile.read(file)
            signalData, samplingFrequency = librosa.load(file, sr=3000)
            # signalData = signalData[:,0]
            # signalData = signal.decimate(signalData, 6)

            spectrum, freqs, t, im = mat_agg.pyplot.specgram(signalData[:], Fs=samplingFrequency)

            temporal_features = spectrum.flatten('F')
            # print(len(spectrum[0]))
            temporal_features = np.pad(temporal_features, (0, 23 * 129 - len(temporal_features)), 'constant')

            mat_agg.pyplot.clf()
            return temporal_features
        except Exception:
            pass

    def create_extracted_features_arr(self):
        print("ran")
        metadata = pd.read_csv('directory.csv')

        for index_num, row in metadata.iterrows():
            file_name = os.path.join(os.getcwd(), 'filtered_samples/', str(row["File name"]))
            # print(file_name)
            # final_class_labels=row["class"]
            data = self.features_extractor(file_name)
            if data is None:
                pass
            else:
                extracted_features.append([data, file_name])

        #print("we did this at least")
        ### add in try except for this later
        """
        if not os.path.exists('extracted_features.csv'):

            


            tmp_arr = []
            for index_num, row in metadata.iterrows():
                file_name = os.path.join(os.getcwd(), 'filtered_samples/', str(row["File name"]))
                # print(file_name)
                # final_class_labels=row["class"]
                data = self.features_extractor(file_name)
                if data is None:
                    pass
                else:
                    extracted_features.append([data, file_name])
            # after extracted_features is made, only add the data values to tmp arr
            for row in extracted_features:
                tmp_arr.append(row[0])
            numpy.savetxt("extracted_features.csv", tmp_arr,
                          delimiter=" ")
        else:
            with open("extracted_features.csv", 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    # print(type(row[0]))
                    np_arr = numpy.fromstring(row[0], sep=' ')
                    extracted_features.append([np_arr])
            i = 0
            for index_num, row in metadata.iterrows():
                file_name = os.path.join(os.getcwd(), 'filtered_samples/', str(row["File name"]))
                extracted_features[i].append(file_name)
                i = i + 1
        """
    extracted_features_df = None
    pca_df = None
    pca_df_list = None
    standardized_data = None
    tsne_df = None
    tsne_df_list = None

    # with a file
    def run_model(self, file_name, extracted_features_df, pca_df, pca_df_list, standardized_data, tsne_df,
                  tsne_df_list):
        '''
        if pca_df == None:
            print("pca_df")
        if pca_df_list == None:
            print("pca_df_list")
        if standardized_data == None:
            print("standardized_data")
        '''
        file = file_name
        filename_list = extracted_features_df['filename'].tolist()
        rownum = filename_list.index(file)
        sound_coords = pca_df.iloc[rownum].tolist()
        # filename_list = extracted_features_df['filename'].tolist()
        '''
        model = TSNE(n_components=2, perplexity=30)
        tsne_dff = model.fit_transform(standardized_data)
        tsne_df = pd.DataFrame(data=tsne_dff)
        tsne_df_list = tsne_df.values
        '''
        rownum2 = filename_list.index(file)
        sound_coords_2 = tsne_df.iloc[rownum2].tolist()

        self.plot_new_point(tsne_df_list, sound_coords_2)

        # matplotlib graph
        # fig = plt.scatter(tsne_df_list[:, 0], tsne_df_list[:, 1], s=1, linewidths=0.5)
        # plt.plot(sound_coords[0], sound_coords[1], 'r*')
        # plt.show()

        # code for plotting?
        # end of code for plotting
        data = []

        for i in range(0, len(pca_df_list)):
            dist = 0
            for j in range(0, len(pca_df_list[0])):
                dist += (pca_df_list[i][j] - sound_coords[j]) ** 2
            data.append([dist, filename_list[i]])
        # print(data)
        panda_df = pd.DataFrame(data, columns=['Distance', 'File Name'])
        panda_df_sorted = panda_df.sort_values(by=['Distance'], ignore_index=True)


        # clear all buttons
        for i in reversed(range(self.vbox.count())):
            self.vbox.itemAt(i).widget().setParent(None)

        #print(panda_df_sorted)
        for i in range(0, min([panda_df_sorted.shape[0],51])):
            name = panda_df_sorted.iloc[i]['File Name']

            obj = QPushButton(name, self)
            text = obj.text()
            obj.clicked.connect(lambda ch, a=i: self.on_button_click(a))
            self.obj_list.append(obj)
            self.vbox.addWidget(obj)

            full_file_path = os.path.join(os.getcwd(), 'filtered_samples/', name)
            # print(full_file_path)
            file_list.append(full_file_path)

    def run_file_model(self):
        global extracted_features_df
        extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'filename'])
        extracted_features_df['filename'] = extracted_features_df['filename'].str.replace(
            os.path.join(os.getcwd(), 'filtered_samples/'), '')
        #print(extracted_features_df)
        # print(extracted_features_df.head(10))

        features_list = extracted_features_df['feature'].tolist()
        #print(features_list)
        stacked_features = np.vstack(features_list)

        global standardized_data
        standardized_data = StandardScaler().fit_transform(stacked_features)
        # print(standardized_data.shape)

        # pca = PCA(n_components=0.85)
        # pca_features = pca.fit_transform(standardized_data)
        # tsne_df = pd.DataFrame(data=pca_features)
        # tsne_df_list = tsne_df.values
        # plot_new_graph(tsne_df_list)

        global tsne_df
        global tsne_df_list
        model = TSNE(n_components=2, perplexity=30)
        tsne_data = model.fit_transform(standardized_data)
        tsne_df = pd.DataFrame(data=tsne_data)
        tsne_df_list = tsne_df.values
        self.plot_new_graph(tsne_df_list)

        global pca_df
        global pca_df_list
        pca = PCA(n_components=0.85)
        pca_features = pca.fit_transform(standardized_data)
        pca_df = pd.DataFrame(data=pca_features)
        pca_df_list = pca_df.values
        inputed_file = ""

    def open_file(self, inputed_file):
        if (folder_loaded):
            global file_list
            file_list = []

            # song_box.delete(0, END)
            # file_path = filedialog.askopenfilename(title="Choose a File", filetypes=(("mp3 Files", "*.mp3 *.wav"),))

            # abbr_file_path = file_path.split('/')[-1].split('.')[0]
            # abbr_file_path = file_path.split('/')[-1]
            # inputed_file = abbr_file_path
            # inputed_file = "ORBIT_Snare_Discovery.wav"
            # should make a progress bar here or smthg to know its working
            # openNewWindow("Working") for some reason it only pops up after the run_model finishes
            self.run_model(inputed_file, extracted_features_df, pca_df, pca_df_list, standardized_data, tsne_df,
                           tsne_df_list)
        else:
            # make a new window pop up saying there is no filtered_samples folder
            # openNewWindow("There is no filtered_samples folder. Press Open Folder")
            print("no folder")

    def play(self, index):
        # sound = song_box.get(ACTIVE)
        print(index)
        sound = file_list[index]
        print(sound)
        pygame.mixer.music.load(sound)
        pygame.mixer.music.play(loops=0)

    def reload_combobox(self):
        #self.combobox.clear()
        self.soundlist = [""]
        for filename in os.listdir("filtered_samples"):
            self.soundlist.append(filename)
        self.combobox.addItems(self.soundlist)


    def plot_new_graph(self, tsne_df_list):
        print("graph")
        # the figure that will contain the plot
        fig = Figure(figsize=(5, 5),
                     dpi=100)
        # adding the subplot
        plot1 = fig.add_subplot(111)
        # plot1.scatter(np.log(self[:, 0]+constant), np.log(self[:, 1]+constant), s=1)
        plot1.scatter(tsne_df_list[:, 0], tsne_df_list[:, 1], s=1)
        plot1.axes.get_xaxis().set_visible(False)
        plot1.axes.get_yaxis().set_visible(False)
        tmp = []
        for val in tsne_df_list[:, 1]:
            if val < 0:
                tmp.append(-np.log(-val))
            elif val > 0:
                tmp.append(np.log(val))
            elif val == 0:
                tmp.append(0)

        # constant = min(self[:,0])
        # plot1.scatter(np.log(self[:, 0]-constant+1), tmp, s=1)
        # plotting the graph
        # plot1.plot()

        # creating the Tkinter canvas
        # containing the Matplotlib figure

        fig.savefig('figure.png', transparent=True)


        self.rightLayout.removeWidget(self.plot)

        pixmapPlot = QPixmap("figure.png")
        self.plot = QLabel(self.pushButton2)
        pixmapPlot_resized = pixmapPlot.scaledToWidth(600)
        self.plot.setPixmap(pixmapPlot_resized)
        self.rightLayout.addWidget(self.plot)

        #canvas = FigureCanvas(fig)  # create canvas
        #self.rightLayout.addWidget(canvas)

    def plot_new_point(self, tsne_df_list, point):
        print("point")
        # the figure that will contain the plot
        fig = Figure(figsize=(5, 5),
                     dpi=100)
        # adding the subplot
        plot1 = fig.add_subplot(111)
        plot1.scatter(tsne_df_list[:, 0], tsne_df_list[:, 1], s=1)
        plot1.scatter(point[0], point[1], color="#FFA500", marker="*", s=50)
        # plotting the graph
        plot1.plot()
        plot1.axes.get_xaxis().set_visible(False)
        plot1.axes.get_yaxis().set_visible(False)
        # creating the Tkinter canvas
        # containing the Matplotlib figure

        #for i in reversed(range(self.rightLayout.count())):
        #    self.rightLayout.itemAt(i).widget().setParent(None)

        fig.savefig('figure.png', transparent=True)
        self.rightLayout.removeWidget(self.plot)

        pixmapPlot = QPixmap("figure.png")
        self.plot = QLabel(self.pushButton2)
        pixmapPlot_resized = pixmapPlot.scaledToWidth(600)
        self.plot.setPixmap(pixmapPlot_resized)
        self.rightLayout.addWidget(self.plot)

    # remove the plot and the song box
    def clear(self):
        global file_list
        self.plot()
        file_list = []
        # clear the buttons

    def plot(self):
        # the figure that will contain the plot
        fig = Figure(figsize=(5, 5),
                     dpi=100)
        # adding the subplot
        plot1 = fig.add_subplot(111)

        # plotting the graph
        plot1.plot()
        plot1.axes.get_xaxis().set_visible(False)
        plot1.axes.get_yaxis().set_visible(False)

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                   master=window)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid(row=1, column=3, rowspan=4, padx=50, pady=10)

stylesheet = """
    MainWindow {
        background-image: url("background.jpg"); 
        background-repeat: no-repeat; 
        background-position: center;
    }
"""

fig = Figure()
ax = fig.add_subplot(111)

ax.axis("off")
ax.patch.set_alpha(0.1)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

