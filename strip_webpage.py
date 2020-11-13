import pandas as pd
import nltk
import selenium
import unicodedata
import re
import platform
from selenium.webdriver.chrome.options import Options
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

class WebPageVectorizer():
    """
    Simple word count vectorizer for web pages
       """

    custom_stopwords = []
    real_words = None
    driver = None
    working_dictionary = ""
    """
    This is where we dump all of our words
    """

    def __init__(self):
        nltk.download('words')
        nltk.download('stopwords')
        self.custom_stopwords = stopwords.words('english')
        self.real_words = set(nltk.corpus.words.words())
        options = Options()
        options.headless = True
        operating_system = platform.system()
        chrome_driver = './drivers/mac_chromedriver86'
        if operating_system == "Linux":
            chrome_driver = './drivers/linux_chromedriver86'
        elif operating_system == "Darwin":
            chrome_driver = './drivers/mac_chromedriver86'
        elif operating_system == "Windows":
            chrome_driver = './drivers/win_chromedriver86.exe'
        self.driver = selenium.webdriver.Chrome(
            options=options,
            executable_path=chrome_driver)

    def __del__(self):
        self.driver.quit()
    
    def add_words(self, text: str):
        self.working_dictionary += str(" " + text)

    @staticmethod
    def remove_accented_chars(text: str):
        """
        remove accents and replaces it with normal boring latin characters
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def print_key_words(self):
        """
        will both print and return a list of keywords in order of frequency 
        """
        vectorizer = CountVectorizer(stop_words=self.custom_stopwords)
        matrix = vectorizer.fit_transform([self.working_dictionary])
        df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()).T
        sorted_words = df.sort_values(by=[0], ascending=False)
        for word in sorted_words.index:
            print(word) 
        print(sorted_words)
        return list(sorted_words.index)

    def strip_webpage(self, url: str) -> str:
        """
        This will strip a webpage for just words

        returns a string
        """
        self.driver.get(url)
        stripper = MLStripper()
        stripper.feed(self.driver.page_source)
        text = stripper.get_data()
        text = re.sub(r'([A-Za-z]*?(\d|\_)+[A-Za-z]*)+', '', text)
        text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in self.real_words or not w.isalpha())
        return text


def main():
    wpv = WebPageVectorizer()

    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/844144/EHS-Director-APAC'))
    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/843659/Service-Engineer-2'))
    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/849210/SR-Data-Applied-Scientist'))

    wpv.print_key_words()

if __name__ == '__main__':
    main()