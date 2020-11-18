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
from selenium.webdriver.support.ui import WebDriverWait
from nltk.corpus import stopwords
from io import StringIO
from html.parser import HTMLParser

class WebpageANalyzer():
    """
    Using a few libraries this has a simple word 
    count vectorizer for web, along with keyword, 
    and clause tools
    """

    custom_stopwords = []
    real_words = None
    driver = None
    working_dictionary = []
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
        self.working_dictionary.append(str(" " + text))

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
        visible_texts = filter(WebpageANalyzer.tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

    def get_keywords(self, text='') -> list:
        """
        Using RAKE extract keywords into a list
        This functino will only return them

        returns a list
        """
        if not bool (text):
            text = ""
            for doc in self.working_dictionary:
                text += (" " + doc)
        self.keyword_rake.extract_keywords_from_text(text)
        return self.keyword_rake.get_ranked_phrases_with_scores()

    def print_keywords(self, text='') -> list:
        """
        Using RAKE extract keywords into a list
        This functino will print them as well as return them

        returns a list
        """
        if not bool (text):
            text = ""
            for doc in self.working_dictionary:
                text += (" " + doc)
        self.keyword_rake.extract_keywords_from_text(text)
        keywords = self.keyword_rake.get_ranked_phrases()
        for word in keywords:
            print(word)
        return keywords

    def get_freq_words(self, text='') -> list:
        """
        Will just return a list of words in order of frequency

        Set text to adjust with a param., default is all text

        returns the list
        """
        if not bool (text):
            text = ""
            for doc in self.working_dictionary:
                text += (" " + doc)
        vectorizer = CountVectorizer(stop_words=self.custom_stopwords)
        matrix = vectorizer.fit_transform([text])
        df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()).T
        sorted_words = df.sort_values(by=[0], ascending=False)
        word_list = []
        for i in range(0, len(sorted_words)):
            word_list.append([sorted_words.index[i], sorted_words.iloc[i, 0]]) 
        return word_list

    def print_freq_words(self, text='') -> list:
        """
        Will both print and return a list of words in order of frequency

        Set text to adjust with a param., default is all text

        returns the list
        """
        if not bool (text):
            text = ""
            for doc in self.working_dictionary:
                text += (" " + doc)
        vectorizer = CountVectorizer(stop_words=self.custom_stopwords)
        matrix = vectorizer.fit_transform([text])
        df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()).T
        sorted_words = df.sort_values(by=[0], ascending=False)
        for i in range(0, len(sorted_words)):
            print(sorted_words.index[i], "(uses:", str(sorted_words.iloc[i, 0]) + ")") 
        print(sorted_words)
        return list(sorted_words.index)

    def strip_webpage(self, url: str, wait=0, method=None, timeout=5) -> str:
        """
        This will strip a webpage for just words

        url is the url

        Wait is the time to let the page load if needed
        method will be some form of webdriverWait if needed
            timeout will get plugged into this if used

        An example of this would be like:
        .. code-block:: python
            method = ec.presence_of_element_located((
                    By.XPATH, ('//div[@id="Postdespacho"]/'
                            '/table[@id="GeneracionXAgente"]')))
            timeout = 10

        The result would end up like this:
        `WebDriverWait(self.driver, timeout).until(method)`


        returns a string
        """
        self.driver.get(url)
        time.sleep(wait)
        try:
            if bool (method):
                WebDriverWait(self.driver, timeout).until(method)
        except Exception as e:
            print(e, "\n Continuing anyway")
        text = WebpageANalyzer.text_from_html(str(self.driver.page_source))
        # som3 w0rd5 w3re n0n3_5ense s0 this cle4n5 th4t up
        text = re.sub(r'([A-Za-z]*?(\d|\_)+[A-Za-z]*)+', '', text)
        # bellow forces it to match words that in the nltk dicitonary (real words more or less)
        #text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in self.real_words or not w.isalpha())
        return text

    def get_analysis_for_keywords(self) -> list:
        """
        This uses a combination of some of the other functions
        This basically takes the RAKE list and cross checks
        it with the frequency list. The more frequent the
        keywords, the better the rank

        This returns a list of keywords 
        """
        all_words = self.get_freq_words()
        set_all_words = {}
        for word in all_words:
            set_all_words[word[0]] = word[1]

        # spacy is smart and knows some basic grammer and we can utilize that too!

        lump = ""
        for chunk in self.get_spacy_doc().noun_chunks:
            lump = (lump + " " + str(chunk))
        for entity in self.get_spacy_doc().ents:
            lump = (lump + " " + str(entity))
        
        special_words = self.get_freq_words(text=lump)

        set_special_words = {}
        for word in special_words:
            set_special_words[word[0]] = word[1]

        intersection_words = set(set_all_words.keys()).intersection(set_special_words.keys())
        for word in intersection_words:
            set_all_words[word] = set_all_words[word] + set_special_words[word]

        # last but not least and probably most important
        docs_keywords = []
        for doc in self.working_dictionary:
            docs_keywords.append(self.get_keywords(text=doc))
        keyword_ranks = []
        keyword_words = []

        keywords = []
        for doc_keyword_list in docs_keywords:
            if not bool(keywords):
                keywords.extend(doc_keyword_list)
            else:
                for i in range(0, len(doc_keyword_list)):
                    keyword = doc_keyword_list[i]
                    if keyword in keywords:
                        for j in range(0, len(keywords)):
                            word = keywords[j]
                            if word[1] == keyword[1]:
                                new_rank = word[0] * keyword[0]
                                word = word[1]
                                keywords[i] = (new_rank, word)
                    else:
                        keywords.append(keyword)
        for i in range(0, len(keywords)):
            word_rank = keywords[i][0]
            words = keywords[i][1].split(' ')
            unique_words = []
            [unique_words.append(word) for word in words if word not in unique_words] 
            # gets rid of nonsense that wasn't filtered
            if len(words) > 5:
                word_rank = 0
                keyword_ranks.append(keywords[i][0])
                keyword_words.append(keywords[i][1])
                continue
            for word in unique_words:
                if word in set_all_words:
                    word_rank *= set_all_words[word] / len(unique_words)
            keyword_ranks.append(word_rank)
            keyword_words.append(keywords[i][1])
        keyword_freq_rank = zip(keyword_ranks, keyword_words)
        return sorted(keyword_freq_rank, key = lambda t: t[0], reverse=True)

    def print_analysis_for_keywords(self) -> list:
        """
        Call the get analysis and prints the list

        Will also return your list
        """
        keywords = self.get_analysis_for_keywords()
        for words in keywords:
            print(words[1], "\t(rank:", (str(words[0]) + ")"))
        return keywords

    def get_spacy_doc(self, doc_number=-1):
        lang_instance = spacy.load('en_core_web_sm')
        if doc_number < 0:
            text = ""
            for doc in self.working_dictionary:
                text += (" " + doc)
            return lang_instance(text)
        else:
            return lang_instance(self.working_dictionary[doc_number])


def main():
    wpa = WebpageANalyzer()

    wpa.add_words(wpa.strip_webpage('https://careers.microsoft.com/us/en/job/844144/EHS-Director-APAC', wait=5))
    wpa.add_words(wpa.strip_webpage('https://careers.microsoft.com/us/en/job/843659/Service-Engineer-2', wait=5))
    wpa.add_words(wpa.strip_webpage('https://careers.microsoft.com/us/en/job/849210/SR-Data-Applied-Scientist', wait=5))

    wpa.print_analysis_for_keywords()

if __name__ == '__main__':
    main()