import pandas as pd
import nltk
import selenium
import unicodedata
import re
import spacy
import platform
import time
from rake_nltk import Rake
from bs4 import BeautifulSoup
from pathlib import Path
from bs4.element import Comment
from selenium.webdriver.chrome.options import Options
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from io import StringIO
from html.parser import HTMLParser

class WebPageVectorizer():
    """
    Simple word count vectorizer for web pages
       """

    custom_stopwords = []
    real_words = None
    driver = None
    working_dictionary = ""
    keyword_rake = Rake()

    """
    This is where we dump all of our words
    """

    def __init__(self):
        nltk.download('words')
        nltk.download('stopwords')
        self.custom_stopwords = stopwords.words('english')
        self.real_words = set(nltk.corpus.words.words())
        self.keyword_rake = Rake(stopwords=self.custom_stopwords)

        options = Options()
        options.headless = True
        operating_system = platform.system()

        full_path = str(__file__)
        full_path = str(Path(full_path).parents[0])

        chrome_driver = '/drivers/mac_chromedriver86'
        if operating_system == "Linux":
            chrome_driver = '/drivers/linux_chromedriver86'
        elif operating_system == "Darwin":
            chrome_driver = '/drivers/mac_chromedriver86'
        elif operating_system == "Windows":
            chrome_driver = '/drivers/win_chromedriver86.exe'
        self.driver = selenium.webdriver.Chrome(
            options=options,
            executable_path=(full_path + chrome_driver))

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

    @staticmethod
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    @staticmethod
    def text_from_html(text: str):
        """
        Strips text off html

        This is taken from here: 
        https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
        """
        soup = BeautifulSoup(text, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(WebPageVectorizer.tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

    def print_keywords(self, text='') -> list:
        """
        Using RAKE extract keywords into a list
        This functino will print them as well as return them

        returns a list
        """
        if not bool (text):
            text = self.working_dictionary
        self.keyword_rake.extract_keywords_from_text(text)
        keywords = self.keyword_rake.get_ranked_phrases()
        for word in keywords:
            print(word)
        return keywords


    def print_freq_words(self, text='') -> list:
        """
        Will both print and return a list of words in order of frequency

        Set text to adjust with a param., default is all text

        returns the list
        """
        if not bool (text):
            text = self.working_dictionary
        vectorizer = CountVectorizer(stop_words=self.custom_stopwords)
        matrix = vectorizer.fit_transform([text])
        df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()).T
        sorted_words = df.sort_values(by=[0], ascending=False)
        for i in range(0, len(sorted_words)):
            print(sorted_words.index[i], "(uses:", str(sorted_words.iloc[i, 0]) + ")") 
        print(sorted_words)
        return list(sorted_words.index)

    def strip_webpage(self, url: str, wait=0) -> str:
        """
        This will strip a webpage for just words

        url is the url
        wait is the time to let the page load if needed

        returns a string
        """
        self.driver.get(url)
        time.sleep(wait)
        text = WebPageVectorizer.text_from_html(str(self.driver.page_source))
        # som3 w0rd5 w3re n0n3_5ense s0 this cle4n5 th4t up
        text = re.sub(r'([A-Za-z]*?(\d|\_)+[A-Za-z]*)+', '', text)
        # bellow forces it to match words that in the nltk dicitonary (real words more or less)
        #text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in self.real_words or not w.isalpha())
        return text

    def get_spacy_doc(self):
        lang_instance = spacy.load('en_core_web_sm')
        return lang_instance(self.working_dictionary)


def main():
    wpv = WebPageVectorizer()

    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/844144/EHS-Director-APAC', wait=5))
    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/843659/Service-Engineer-2', wait=5))
    wpv.add_words(wpv.strip_webpage('https://careers.microsoft.com/us/en/job/849210/SR-Data-Applied-Scientist', wait=5))

    wpv.print_freq_words()

    # spacy is smart and knows some basic grammer and we can utilize that too!

    lump = ""
    for chunk in wpv.get_spacy_doc().noun_chunks:
        lump = (lump + " " + str(chunk))

    for entity in wpv.get_spacy_doc().ents:
        lump = (lump + " " + str(entity))
    
    wpv.print_freq_words(text=lump)
    
    # last but not least and probably most important
    wpv.print_keywords()

if __name__ == '__main__':
    main()