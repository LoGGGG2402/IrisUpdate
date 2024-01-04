from tkinter import *
from tkinter import filedialog

from PIL import ImageTk
from PIL import Image as PilImage

import segmentation as se

import cv2

from feature_extraction import encode_iris
from matching import matching

img_bg = "pink"


class GUI:
    can_segment = False
    image1_path = ""
    image2_path = ""
    iris_code1 = None
    iris_code2 = None
    time = 0

    def __init__(self, master):
        self.master = master
        master.geometry("1000x600")
        master.title("Iris Recognition System")

        # Image frames on the left side
        self.image_frame = Frame(master, bg=img_bg)
        self.image_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # Menu on the right side
        self.menu_frame = Frame(master, background='gray')
        self.menu_frame.pack(side=RIGHT, fill=Y)

        # Buttons: select image 1 and 2, segmentation, normalization, feature extraction, check similarity, exit
        self.select_image1_button = Button(self.menu_frame, text="Select Image 1", command=self.select_image1)
        self.select_image1_button.pack(fill=X, padx=10, pady=10)
        self.select_image2_button = Button(self.menu_frame, text="Select Image 2", command=self.select_image2)
        self.select_image2_button.pack(fill=X, padx=10, pady=10)
        self.segmentation_button = Button(self.menu_frame, text="Segmentation", command=self.segmentation, bg='white')
        self.segmentation_button.pack(fill=X, padx=10, pady=10)
        self.feature_extraction_button = Button(self.menu_frame, text="Feature Extraction",
                                                command=self.feature_extraction,
                                                bg='white')
        self.feature_extraction_button.pack(fill=X, padx=10, pady=10)
        self.check_similarity_button = Button(self.menu_frame, text="Check Similarity", command=self.check_similarity,
                                              bg='white')
        self.check_similarity_button.pack(fill=X, padx=10, pady=10)
        self.exit_button = Button(self.menu_frame, text="Exit", command=master.quit, bg='white')

        # 4 images with 2 rows and 2 columns each having a label and size of 320x240
        # Input image1
        self.input_image1 = Canvas(self.image_frame, bg='white', height=240, width=320)
        self.input_image1.grid(row=0, column=0, padx=10, pady=10)

        # Label below input image1
        self.input_label1 = Label(self.image_frame, text="Input Label 1", bg=img_bg)
        self.input_label1.grid(row=1, column=0, padx=10, pady=(0, 10))

        # Input image2
        self.input_image2 = Canvas(self.image_frame, bg='white', height=240, width=320)
        self.input_image2.grid(row=2, column=0, padx=10, pady=10)

        # Label below input image2
        self.input_label2 = Label(self.image_frame, text="Input Label 2", bg=img_bg)
        self.input_label2.grid(row=3, column=0, padx=10, pady=(0, 10))

        # Segmented image
        self.segmented_image1 = Canvas(self.image_frame, bg='white', height=240, width=320)
        self.segmented_image1.grid(row=0, column=1, padx=10, pady=10)

        # Label below segmented image
        self.segmented_label1 = Label(self.image_frame, text="Segmented Label 1", bg=img_bg)
        self.segmented_label1.grid(row=1, column=1, padx=10, pady=(0, 10))

        # Segmented image
        self.segmented_image2 = Canvas(self.image_frame, bg='white', height=240, width=320)
        self.segmented_image2.grid(row=2, column=1, padx=10, pady=10)

        # Label below segmented image
        self.segmented_label2 = Label(self.image_frame, text="Segmented Label 2", bg=img_bg)
        self.segmented_label2.grid(row=3, column=1, padx=10, pady=(0, 10))

        # Result Frame
        self.result_frame = Frame(self.image_frame, bg="white", height=240, width=200)
        self.result_frame.grid(row=0, column=3, padx=10, pady=10, rowspan=2)

        # Label of result frame
        self.result_label = Label(self.result_frame, text="Result", bg="white")

        # Similarity
        # Label of similarity
        self.similarity_label = Label(self.result_frame, text="Similarity", bg="white")
        # Result of similarity
        self.similarity_result = Label(self.result_frame, text="0%", bg="white")

        # Time
        # Label of time
        self.time_label = Label(self.result_frame, text="Time", bg="white")
        # Result of time
        self.time_result = Label(self.result_frame, text="0s", bg="white")

        # Match
        # Label of match
        self.match_label = Label(self.result_frame, text="Match", bg="white")
        # Result of match
        self.match_result = Label(self.result_frame, text="False", bg="white")

        # Grid
        self.result_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        self.similarity_label.grid(row=1, column=0, padx=10, pady=10)
        self.similarity_result.grid(row=1, column=1, padx=10, pady=10)
        self.time_label.grid(row=2, column=0, padx=10, pady=10)
        self.time_result.grid(row=2, column=1, padx=10, pady=10)
        self.match_label.grid(row=3, column=0, padx=10, pady=10)
        self.match_result.grid(row=3, column=1, padx=10, pady=10)

    def select_image1(self):
        file_path = filedialog.askopenfilename(title="Select file")
        if file_path:
            if file_path.endswith(".jpg") or file_path.endswith(".bmp"):
                self.image1_path = file_path
                image1 = PilImage.open(self.image1_path)
                image1 = image1.resize((320, 240), PilImage.Resampling.LANCZOS)
                image1 = ImageTk.PhotoImage(image1)
                self.input_image1.background = image1
                self.input_image1.create_image(0, 0, anchor=NW, image=image1)
                img_name = self.image1_path.split("/")[-1]
                self.input_label1.config(text=img_name)
                self.can_segment = False
                self.segmentation_button.config(bg='white')
                self.segmentation_button.config(text="Segmentation")
                self.segmentation_button.config(state=NORMAL)

    def select_image2(self):
        file_path = filedialog.askopenfilename(title="Select file")
        if file_path:
            if file_path.endswith(".jpg") or file_path.endswith(".bmp"):
                self.image2_path = file_path
                image2 = PilImage.open(self.image2_path)
                image2 = image2.resize((320, 240), PilImage.Resampling.LANCZOS)
                image2 = ImageTk.PhotoImage(image2)
                self.input_image2.background = image2
                self.input_image2.create_image(0, 0, anchor=NW, image=image2)
                img_name = self.image2_path.split("/")[-1]
                self.input_label2.config(text=img_name)
                self.can_segment = False
                self.segmentation_button.config(bg='white')
                self.segmentation_button.config(text="Segmentation")
                self.segmentation_button.config(state=NORMAL)

    def segmentation(self):
        segmented1 = se.iris_segmentation(img_path=self.image1_path)
        segmented2 = se.iris_segmentation(img_path=self.image2_path)
        if segmented1 is not None:
            segmented_iris, inner_center, inner_radius, outer_center, outer_radius = segmented1
            img_cv = segmented_iris
            img_cv = cv2.circle(img_cv, (inner_center[0], inner_center[1]), inner_radius, (0, 255, 0), 2)
            img_cv = cv2.circle(img_cv, (outer_center[0], outer_center[1]), outer_radius, (0, 255, 0), 2)
            img_cv = PilImage.fromarray(img_cv)
            img_cv = img_cv.resize((320, 240), PilImage.Resampling.LANCZOS)
            img_cv = ImageTk.PhotoImage(img_cv)
            self.segmented_image1.background = img_cv
            self.segmented_image1.create_image(0, 0, anchor=NW, image=img_cv)
            self.segmented_label1.config(text="Segmented Image 1")
        if segmented2 is not None:
            segmented_iris, inner_center, inner_radius, outer_center, outer_radius = segmented2
            img_cv = segmented_iris
            img_cv = cv2.circle(img_cv, (inner_center[0], inner_center[1]), inner_radius, (0, 255, 0), 2)
            img_cv = cv2.circle(img_cv, (outer_center[0], outer_center[1]), outer_radius, (0, 255, 0), 2)
            img_cv = PilImage.fromarray(img_cv)
            img_cv = img_cv.resize((320, 240), PilImage.Resampling.LANCZOS)
            img_cv = ImageTk.PhotoImage(img_cv)
            self.segmented_image2.background = img_cv
            self.segmented_image2.create_image(0, 0, anchor=NW, image=img_cv)
            self.segmented_label2.config(text="Segmented Image 2")
        if segmented1 is not None and segmented2 is not None:
            self.can_segment = True
            self.segmentation_button.config(bg='green')
            self.segmentation_button.config(text="Segmentation Done")
            self.segmentation_button.config(state=DISABLED)
        else:
            self.can_segment = False
            self.segmentation_button.config(bg='white')
            self.segmentation_button.config(text="Segmentation")
            self.segmentation_button.config(state=DISABLED)

    def feature_extraction(self):
        if self.can_segment is False:
            return None
        t1 = cv2.getTickCount()
        # Notifiy the user that the feature extraction is in progress
        self.feature_extraction_button.config(bg='yellow')
        self.feature_extraction_button.config(text="Feature Extraction in Progress")
        self.feature_extraction_button.config(state=DISABLED)

        self.iris_code1 = encode_iris(image_path=self.image1_path)
        self.iris_code2 = encode_iris(image_path=self.image2_path)
        t2 = cv2.getTickCount()
        # Notifiy the user that the feature extraction is done
        self.feature_extraction_button.config(bg='white')
        self.feature_extraction_button.config(text="Feature Extraction")
        self.feature_extraction_button.config(state=NORMAL)
        self.time = (t2 - t1) / cv2.getTickFrequency()

    def check_similarity(self):
        if self.iris_code1 is None or self.iris_code2 is None:
            return
        else:
            match, simutality = matching(self.iris_code1, self.iris_code2)
            if match:
                self.match_result.config(text="True")
            else:
                self.match_result.config(text="False")
            # shorten the number of digits after the decimal point to 2
            self.time = round(self.time, 2)
            simutality = round(simutality * 100, 2)
            self.similarity_result.config(text=str(simutality) + "%")
            self.time_result.config(text=str(self.time) + "s")


if __name__ == '__main__':
    root = Tk()
    gui = GUI(root)
    root.mainloop()
