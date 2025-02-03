from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
def get_tfidf_labels(class1 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    , class2 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']):
    newsgroups_all = fetch_20newsgroups(subset='all', remove = ('headers', 'footers', 'quotes'), categories = class1 + class2)
    
    vectorizer = TfidfVectorizer(min_df = 3, stop_words = 'english')
    all_tfidf = vectorizer.fit_transform(newsgroups_all.data)

    all_labels = newsgroups_all.target
    """
    Answer for problem1
    """
    return all_tfidf, all_labels

if __name__ == "__main__":
    class1 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    class2 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    all_tfidf, _ = get_tfidf_labels(class1, class2)
    print(all_tfidf.shape)