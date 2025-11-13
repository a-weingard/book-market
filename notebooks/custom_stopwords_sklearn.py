from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

combined_stopwords = sorted(list(ENGLISH_STOP_WORDS.union([
    # Words with contractions (don't, we'll)
    'don', 't', 's', 'll', 're', 've', 'm', 'd', 'did', 'didn', 'does', 'doesn', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn', 'shouldn', 'wouldn', 'couldn', 'doing',  
                        
    # Allgemeine Buchbegriffe
    'book', 'novel', 'story', 'tale', 'author', 'writer', 'read', 'reader', 'reading', 'writes',
    
    # Häufige Figuren oder Szenenwörter
    'character', 'characters', 'person', 'people', 
    #'friend', 'family', 'life', 'love', 'relationships', 'feelings', 'emotion', 'human', 'man', 'woman', 'girl', 'boy',

    # Plot-Vokabular
    'scene', 'scenes', 'event', 'events', 'plot', 'beginning', 'ending', 'start', 'finish',
    'chapter', 'chapters', 'pages', 'part', 'volume', 'series',

    # Bewertungen / generische Begriffe
    'great', 'good', 'bad', 'amazing', 'interesting', 'boring', 'funny', 'dark', 'beautiful',
    'perfect', 'favorite', 'best', 'worst',

    # Zeit/Ort
    'time', 'day', 'night', 'place', 'country', 'year', 'years','world', 'way',
    # 'city', 'town', 

    # Sprache
    'english', 'translated', 'edition', 'language',

    # Genre-Begriffe
    'fiction', 'nonfiction', 'mystery', 'romance', 'thriller', 'fantasy', 'historical', 'drama', 'comic', 'comics',

    # Weitere häufige sinnentleerte Begriffe
    'one', 'two', 'another', 'someone', 'something', 'things', 'anything', 'everything', 'away', 'just', 'kg', 'km',

    # Verben
    'want', 'need', 'know', 'like', 'return', 'think', 'meet', 'take', 'come', 'take', 'going', 'leave', 'do', 'live', 'go', 'find', 'tell',
    'wants', 'needs', 'knows', 'likes', 'returns', 'thinks', 'meets', 'takes', 'comes', 'takes', 'going', 'leaves', 'does', 'lives', 'finds', 'tells',

    # Adjektive
    'new', 'old', 'young', 'short', 'long', 'high'
])))



