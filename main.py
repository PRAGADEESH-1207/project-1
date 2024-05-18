import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserIconView
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.spinner import MDSpinner
import tensorflow as tf
import numpy as np
import pandas as pd

class RootLayout(BoxLayout):
    pass

class HomePage(BoxLayout):
    pass

class AboutPage(BoxLayout):
    pass

class PredictionPage(BoxLayout):
    pass

class PredictionApp(App):
    def build(self):
        self.title = "FRUITS & VEGETABLES RECOGNITION SYSTEM"
        self.root_layout = RootLayout()
        self.home_page()
        return self.root_layout

    def on_sidebar_change(self, spinner, text):
        content = self.root_layout.ids.content
        content.clear_widgets()
        if text == 'Home':
            content.add_widget(HomePage())
        elif text == 'About Project':
            content.add_widget(AboutPage())
        elif text == 'Prediction':
            content.add_widget(PredictionPage())

    def home_page(self):
        content = self.root_layout.ids.content
        content.clear_widgets()
        content.add_widget(HomePage())

    def predict(self):
        content = self.root_layout.ids.content
        file_chooser = content.ids.file_chooser
        prediction_result = content.ids.prediction_result

        if not file_chooser.selection:
            prediction_result.text = "Please select an image."
            return

        test_image_path = file_chooser.selection[0]
        model = tf.keras.models.load_model("trained_model.h5")
        image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        predictions = model.predict(input_arr)
        predicted_index = np.argmax(predictions)

        with open("labels.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        predicted_item = labels[predicted_index]
        market_price = self.get_market_price(predicted_item)
        prediction_result.text = f"Model is Predicting it's a {predicted_item}. Market price: {market_price} Rs"

    def get_market_price(self, predicted_item):
        excel_file = 'pricelist2.xlsx'
        df = pd.read_excel(excel_file)
        price = df.loc[df['vegatables and fruits '] == predicted_item, 'price'].values
        return price[0] if len(price) > 0 else "Not available"

if __name__ == "__main__":
    PredictionApp().run()
