from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import rdflib
import locale
_ = locale.setlocale(locale.LC_ALL, '')
import numpy as np
import os, json
from joblib import dump, load
import logging
import csv
import random
from string import ascii_letters
from urllib.parse import urlparse
import requests, pickle
from sklearn.feature_extraction.text import strip_accents_ascii

# ! pip install - U sentence - transformers
from sentence_transformers import SentenceTransformer
"""
# ! pip install transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
"""

# predicates_dict['instance of']
# example: for x in graph.objects(entities_dict['carl foreman'], predicates_dict['instance of']):
#   for l in graph.objects(rdflib.term.URIRef(x), label_pred):
#     print(l)

#  - %(message)s
logging.basicConfig(format='=> %(asctime)s - RDFQueries', datefmt='%H:%M:%S')
logger = logging.getLogger('__name__')
level = logging.DEBUG
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)

# prefixes used in the graph
WD = Namespace('http://www.wikidata.org/entity/')
WDT = Namespace('http://www.wikidata.org/prop/direct/')
SCHEMA = Namespace('http://schema.org/')
DDIS = Namespace('http://ddis.ch/atai/')

label_pred = rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#label')
imdb_id = WDT.P345
image = WDT.P18
genre_pred = WDT.P136
type_pred = WDT.P31
movie_obj = WD.Q11424

only_letters = set(ascii_letters + ' ')

API_URL = "https://api-inference.huggingface.co/models/satvikag/chatbot"
headers = {"Authorization": f"Bearer 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"}

def query(q):
    response = requests.post(API_URL, headers=headers, json={"inputs": {"text": q}}).json()
    logger.debug("Query response: ", response)
    if 'generated_text' in response:
        return response['generated_text']
    else:
        return 'No idea.'

def normalize_text(text):
    t = text.replace(" – ", " - ")
    t = strip_accents_ascii(t.lower())
    t = t.replace('?', ' ')
    t = t.replace('.', ' ')
    t = t.replace(',', ' ')
    t = t.replace('\n', ' ')
    t = ' '.join(t.split())
    return ' ' + t + ' '

templates = {'KB':['The answer found in the graph is: ', 'According to the knowledge graph: ', 'I think the answer is '], 
             'emb':['The answer found using embeddings is: ', 'The answer suggested by embeddings: ',
                    'Some other answers found using the embeddings: '],
             'no':['Sorry, I could not find the information you are loking for. Can you paraphrase your question?', 
                   'The information could not be found.', 'Sorry, can you rephrase your question?'],
             'suggest':['Some %s similar to those are: '], # !!!!!!!!!!!!!!!!
             'image':['Here is a %s image from the movie %s (%s):\n%s', 
                      'Here is an image of %s from the movie %s (%s):\n%s',
                      'Here is an image of %s:\n%s'],
             'greet':['Hey!', 'Hello', 'Hi :)'],
             'recommend': [('You can check out the movies ', ' with the genres ', ' which I have found using the graph embeddings.\n'),
                           'You can check out the movie %s found using the %s. Its genre is %s.',
                           'You might be interested in this movie: %s that I found in the %s, with the genre %s.',
                           'Here is the imdb page of the movie: imdb:%s',
                            'See the imdb page here: imdb:%s.',
                            '(imdb:%s)']}
sentence_types = ['full', 'only_obj']

class RDFQueries(object):
    def __init__(self, path):   # path : path to the data folder.
        logger.info('Starting initialization')
        self.path = path

        self.label_pred = rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#label')

        # self.graph = rdflib.Graph()
        # self.graph.parse(os.path.join(self.path, '14_graph.nt'), format='turtle')
        with open(os.path.join(self.path, 'graph.pkl'), 'rb') as file: 
            self.graph = pickle.load(file)
        logger.info('Parse Graph Done')

        # load the sentence transformer
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info('Sentence Embeddings model loaded')
        
        self.load_metadata()
        logger.info('Meta Data are loaded')
        
        """
        # load the NER model
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        logger.info('NER model loaded')
        """
        
        logger.info('READY')

# ************************************************ANSWER*************************************************

    def answer(self, q, k=3):
        try:
            triples, embeddings_triples = self.extract(q)
        except:
            logger.debug(logging.exception("EXCEPTION IN EXTRACT!"))
        if triples == None:
            return query(q)
        if type(triples) == str:
            return triples
        ans = ''
        use_other = False
        if len(triples) > 0:
            ans += templates['KB'][random.randint(0,1)] + self.construct_answer(triples, type=sentence_types[random.randint(0,1)]) + '. \n'
            use_other = True
            for triple in triples:
                for emb_triple in embeddings_triples:
                    if triple[:2] == emb_triple[:2]:
                        emb_triple[2] = list(set(emb_triple[2]) - set(triple[2]))

        if len(embeddings_triples) > 0:
            if use_other:
                ans += templates['emb'][2] + self.construct_answer(embeddings_triples, type=sentence_types[1]) + '\n'
            else:
                ans += templates['emb'][random.randint(0,1)] + self.construct_answer(embeddings_triples, type=sentence_types[random.randint(0,1)]) + '. \n'
        if len(triples) <= 0 and len(embeddings_triples) <= 0:
            # ans = templates['no'][random.randint(0,2)]
            return query(q)
        return ans

    @staticmethod
    # type can be full, only_obj ...
    def construct_answer(triples, type='full'):
        def mult_obj(objects):
            ans = ''
            for obj in objects[:-1]:
                ans += ' %s,'%obj
                ans = ans[:-1]
            ans += ' and %s.'%objects[-1]
            return ans

        ans = ''
        if type=='full':
            for triple in triples:
                objects = triple[2]
                mult = ''
                if len(objects) > 1:
                    mult = 's'
                ans = 'The %s%s of %s '%(triple[1], mult, triple[0])
                if len(objects) == 1:
                    ans += 'is %s.'%objects[0]
                else:
                    ans += 'are'
                    ans += mult_obj(objects)
            return ans
        elif type=='only_obj':
            triple = triples[0]
            objects = triple[2]
            mult = ''
            if len(objects) == 1:
                return objects[0]
            else:
                return mult_obj(objects)
        else:
            triple = triples[0]
            objects = triple[2]
            mult = ''
            if len(objects) == 1:
                return objects[0]
            else:
                return mult_obj(objects) 

    def get_imdb_id(self, ent):
        for s, p, o in self.graph.triples((ent, imdb_id, None)):
            return str(o)
        return None

    # all these should be strings
    def get_image_from_imdb_id(self, movie=None, im_type=None, cast=None, only_person=True):
        ims = self.images_dict.copy()
        if movie != None:
            ims = [d for d in ims if movie in d['movie']]
        if im_type != None:
            ims = [d for d in ims if d['type'] == im_type]
        if cast != None:
            if only_person:
                ims = [d for d in ims if cast in d['cast'] and len(d['cast']) == 1]
            else:
                ims = [d for d in ims if cast in d['cast'] and len(d['cast']) > 1]
        return ims

    def answer_image(self, ent):
        id = self.get_imdb_id(ent)
        if id != None and id.startswith('tt'):  # we are looking for a movie img
            ims = self.get_image_from_imdb_id(movie=id)
            if len(ims) <= 0:
                return None, None
            im = ims[random.randint(0,len(ims)-1)]
            im_type = im['type']
            return templates['image'][0]%(im_type, self.ent2lbl[ent].title(), 
                                'imdb:' + id, 'image:' + im['img'][:-4])
        if id != None and id.startswith('nm'):
            ims = self.get_image_from_imdb_id(cast=id)
            if len(ims) <= 0:
                return None, None
            im = ims[random.randint(0,len(ims)-1)]
            if len(im['movie']) > 0:
                i = 0
                movie_id = None
                while movie_id == None and i < len(im['movie']):
                    id_lit = rdflib.term.Literal(im['movie'][i], datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#string'))
                    for s, p, o in self.graph.triples((None, imdb_id, id_lit)):
                        movie_id = im['movie'][i]
                        movie_name = self.ent2lbl[s]
                    i += 1
                if movie_id == None:
                    return templates['image'][2]%(self.ent2lbl[ent], 'image:' + im['img'][:-4])
                return templates['image'][1]%(self.ent2lbl[ent].title(), movie_name.title(), 
                                            'imdb:' + movie_id, 'image:' + im['img'][:-4])
            else:
                return templates['image'][2]%(self.ent2lbl[ent], 'image:' + im['img'][:-4])
        else:
            return None

    def answer_recommend(self, entities):
        graph_emb_obj = []
        sentence_emb_obj = []
        ans = 'Okay, so '
        genres_init = []
        names_init = []
        for ent in entities:
            genre = None
            names_init.append(self.ent2lbl[ent])
            for s, p, o in self.graph.triples((ent, genre_pred, None)):
                genre = self.ent2lbl[o][1:-1]
                genres_init.append(genre)

        # First using graph embeddings
        embeddings = [self.entities_graph_embeddings[self.ent2id[ent]] for ent in entities]
        lhs = np.mean(embeddings, axis=0)
        obj_ids, _ = self.k_neighbors(lhs, self.entities_graph_embeddings, k=10)
        for i in obj_ids:
            obj = self.id2ent[i]
            if self.ent2lbl[obj] not in names_init:
                genre = None
                for s, p, o in self.graph.triples((obj, genre_pred, None)):
                    genre = self.ent2lbl[o][1:-1]
                for s, p, o in self.graph.triples((obj, type_pred, None)):
                    if o == movie_obj and genre in genres_init: # ??????????????????***
                        graph_emb_obj.append((self.ent2lbl[obj][1:-1].title(), genre, self.get_imdb_id(obj), obj))
                
        if len(graph_emb_obj) == 1:
            ans += 'you can check out the movie %s found using the graph embeddings. \
            Its genre is %s.'%(graph_emb_obj[0][0], graph_emb_obj[0][1])
            id = graph_emb_obj[0][2]
            if id != None and id.startswith('tt'):
                ans += templates['recommend'][random.randint(3,5)]%id
            # if str(graph_emb_obj[0][3]) in plots:
            #    ans += '\nAlso read the plot here: \n %s'%plots[graph_emb_obj[0][3]]
            return ans, ''
        elif len(graph_emb_obj) > 1:
            names = []
            genres = []
            for movie in graph_emb_obj:
                name = movie[0]
                id = movie[2]
                if id != None and id.startswith('tt'):
                    name += ' (imdb:%s), '%id
                names.append(name)
                genres.append(movie[1])
            ans = templates['recommend'][0][0]
            for name in names:
                ans += name
            ans = ans[:-2] + templates['recommend'][0][1]
            for genre in genres:
                ans += genre + ', '
            ans = ans[:-2] + templates['recommend'][0][2]
            return ans, ''

        # Using Sentence Embeddings
        lbls = [self.ent2lbl[ent] for ent in entities]
        embeddings = self.sentence_model.encode(lbls)
        lhs = np.mean(embeddings)
        obj_ids, _ = self.k_neighbors(lhs, self.entities_embeddings, k=10)
        for i in obj_ids:
            objects = self.entities_dict[self.entities_names[i]]
            found = False
            i = 0
            while not found and i < len(objects):
                obj = objects[i]
                if obj in self.ent2lbl and self.ent2lbl[obj] not in names_init:
                    genre = None
                    for s, p, o in self.graph.triples((obj, genre_pred, None)):
                        genre = self.ent2lbl[o][1:-1]
                    for s, p, o in self.graph.triples((obj, type_pred, None)):
                        if o == movie_obj and genre in genres_init: # ??????????????????***
                            sentence_emb_obj.append((self.ent2lbl[obj].title(), genre, self.get_imdb_id(obj)))
                            found = True
                i += 1
        
        if len(sentence_emb_obj) == 1:
            ans += templates['recommend'][random.randint(1,2)]%(sentence_emb_obj[0][0], 'Knowledge Base', 
                                                                sentence_emb_obj[0][1])
            id = sentence_emb_obj[0][2]
            if id != None and id.startswith('tt'):
                ans += templates['recommend'][random.randint(3,5)]%id
            #if str(graph_emb_obj[0][3]) in plots:
            #     ans += '\nAlso read the plot here: \n %s'%plots[graph_emb_obj[0][3]]
            return ans, ''
        elif len(sentence_emb_obj) > 1:
            names = []
            genres = []
            for movie in graph_emb_obj:
                name = movie[0]
                id = movie[2]
                if id != None and id.startswith('tt'):
                    name += ' (imdb:%s), '%id
                names.append(name)
                genres.append(movie[1])
            ans = templates['recommend'][0][0]
            for name in names:
                ans += name
            ans = ans[:-2] + templates['recommend'][0][1]
            for genre in genres:
                ans += genre + ', '
            ans = ans[:-2] + templates['recommend'][0][2]
            return ans, ''
        return '', ''

# ************************************************EXTRACT****************************************************
    def extract(self, q):
        try:
            ne, pred = self.get_named_entities_2(q)
        except:
            logger.debug(logging.exception("EXCEPTION IN GET NAMED ENTITIES!"))
        if pred == 'greet':
            return templates['greet'][random.randint(0,2)], ''
        if len(ne) <= 0 or np.all([len(e) <= 2 for e in ne]):
            return query(q), ''

        embeddings_ent = self.sentence_model.encode(ne)
        triples = []
        embeddings_triples = []
        entities = []

        if pred == 'recommend':
            for emb in embeddings_ent:
                ent_ids = self.k_neighbors(emb, self.entities_embeddings, k=10)
                ent_id = 0
                found = False
                while not found and ent_id < len(ent_ids[0]):
                    i = 0
                    ents = self.entities_dict[self.entities_names[ent_ids[0][ent_id]]]
                    while not found and i < len(ents):
                        ent = ents[i]
                        for s, p, o in self.graph.triples((ent, type_pred, None)):
                            if o == movie_obj: # ??????????????????***
                                entities.append(ent)
                                found = True
                        i += 1
                    ent_id += 1
            if len(entities) > 0:
                return self.answer_recommend(entities)
            else:
                return 'Sorry, I could not find those movies in the KG. ', ''

        elif pred == image:
            ne_id = 0
            found = False
            while not found and ne_id < len(embeddings_ent):
                emb = embeddings_ent[ne_id]
                ent_ids = self.k_neighbors(emb, self.entities_embeddings, k=10)
                ent_id = 0
                while not found and ent_id < len(ent_ids[0]):
                    i = 0
                    ents = self.entities_dict[self.entities_names[ent_ids[0][ent_id]]]
                    while not found and i < len(ents):
                        ent = ents[i]
                        ans = self.answer_image(ent)
                        if ans != None:
                            return ans, ''
                        i += 1
                    ent_id += 1
                ne_id += 1
            return 'Sorry, could not find the movie or actor you are asking for. ', ''

        else:  
            entities = [self.entities_dict[ent_name][0] for ent_name in ne]
            for ent in entities:
                # check if can use embeddings!
                if pred in self.pred2id and ent in self.ent2id:
                    triple = self.get_objects_embeddings(ent, pred)
                    if len(triple) > 0 and len(triple[2]) > 0:
                        embeddings_triples.append(triple)
                triple = self.get_objects(ent, pred)
                if len(triple) > 0 and len(triple[2]) > 0:
                    triples.append(triple)

        return triples, embeddings_triples

    def get_predicate(self, question):
        # check if they just do small talk
        emb = self.sentence_model.encode([question])[0]
        n, _ = self.k_neighbors(emb, np.concatenate([self.predicates_embeddings, self.greet_emb.reshape(1,-1)], axis=0), k=1)
        if n == len(self.predicates_embeddings):
            return 'greet', question

        # first check for look like or looks like
        words = ''.join(l for l in question if l in only_letters)
        if 'look like' in words:
            return image, 'look like'
        if 'looks like' in words:
            return image, 'looks like'
        if 'show' in words:
            return image, 'show'
        if 'picture' in words:
            return image, 'picture'

        # then check if recommendation question
        if 'recommend' in words:
            return 'recommend', 'recommend'
        if 'similar' in words:
            return 'recommend', 'similar'
        if 'suggest' in words:
            return 'recommend', 'suggest'

        # else get most probable predicate
        q = question.split()
        embeddings_pred = self.sentence_model.encode(q)
        pred = (None, 100000, None)
        included = 0
        for i, emb in enumerate(embeddings_pred):
            [p, d] = self.k_neighbors(emb, self.predicates_embeddings, 1)
            if self.predicates_names[p[0]] in question:
                if len(self.predicates_names[p[0]]) > included:
                    included = len(self.predicates_names[p[0]])
                    pred = (p[0], d[0], q[i])
            elif included <= 0:
                if d < pred[1]:
                    pred = (p[0], d[0], q[i])
        predicates_uris = [self.predicates_dict[self.predicates_names[pred[0]]][0]]
        return predicates_uris[0], pred[2]

    def get_objects(self, entity, pred):
        objects = []
        predicate = self.graph.objects(pred, RDFS.label)
        predicate = str(list(predicate)[0])
        for s, p, o in self.graph.triples((entity, pred, None)):
            if o in self.ent2lbl:
                objects.append(self.ent2lbl[o][1:-1].title())
            elif isinstance(o, Literal):
                objects.append(str(o)[1:-1].title())
            else:
                print('WTF IF HAPPENING IN get_objects ????')
        return [self.ent2lbl[entity][1:-1].title(), predicate, objects]

    def get_objects_embeddings(self, entity, pred):
        ent_emb = self.entities_graph_embeddings[self.ent2id[entity]]
        pred_emb = self.predicates_graph_embeddings[self.pred2id[pred]]
        lhs = ent_emb + pred_emb
        obj_ids, _ = self.k_neighbors(lhs, self.entities_graph_embeddings, k=3)
        predicate = self.graph.objects(pred, RDFS.label)
        predicate = str(list(predicate)[0])
        objects = []
        for i in obj_ids:
            objects.append(self.ent2lbl[self.id2ent[i]][1:-1].title())
        return [self.ent2lbl[entity][1:-1].title(), predicate, objects]

    def ner(self, q):
        # idea is, first check all the words to see if they are in entities list.
        ne = []
        if len(ne) == 0:
            txt = q.replace(':', ' ')
            txt = txt.replace('-', ' ')
            if txt[-1] != '?':
                txt += '?'
            ner_results = self.nlp(txt)
            name = None
            try:
                for entry in ner_results:
                    if entry['entity'][0] == 'B':
                        if name != None:
                            ne.append(name.replace(' ##', ''))
                        name = entry['word']
                    elif entry['entity'][0] == 'I':
                        name += ' ' + entry['word']
            except:
                logger.debug(logger.debug('Error in ner function'))
                return []

            if name != None:
                ne.append(name.replace(' ##', ''))
        return ne

    def get_named_entities_2(self, q):
        ne = []
        text = normalize_text(q)
        pred, pred_word = self.get_predicate(text)
        if pred == 'greet':
            return [], pred
        text = text.replace(pred_word, '')
        text = normalize_text(text)
        indices = list(np.where([word in text for word in self.entities_names])[0])
        indices2 = indices.copy()
        for i in indices:
            for j in indices:
                word = self.entities_names[i]
                sub = self.entities_names[j]
                if i != j and sub in word and j in indices2:
                    indices2.remove(j)
        for i in indices2:
            ne.append(self.entities_names[i])
        arr1inds = np.array([len(word) for word in ne]).argsort()
        ne = np.array(ne)[arr1inds[::-1]]
        # if len(ne) <= 0 or np.all([len(e) <= 2 for e in ne]):
        #    ne = self.ner(q)
        if pred == 'recommend':
            return ne, pred
        return ne[:1], pred

    # similarity function
    def k_neighbors(self, query, embeddings, k=1):
        # compute distances
        distances = np.linalg.norm(embeddings - query, axis = 1)
        # select indices of vectors having the lowest distances from the query vector (sorted!)
        neighbors = np.argpartition(distances, range(0, k))[:k]
        return [neighbors, distances[neighbors]]

    def load_metadata(self):
        # load the dictionaries
        self.predicates_dict = load(os.path.join(self.path, 'processed/predicates_dict.joblib'))
        self.entities_dict = load(os.path.join(self.path, 'processed/entities_dict.joblib'))

        self.predicates_names = list(self.predicates_dict.keys())
        self.entities_names = list(self.entities_dict.keys())

        # WOAAAAAAAAAAAAH
        self.ent2lbl = {}
        for key, value in list(self.entities_dict.items()):
            for val in value:
                self.ent2lbl[val] = key

        self.pred2lbl = {}
        for key, value in list(self.predicates_dict.items()):
            for val in value:
                self.pred2lbl[val] = key

        # For graph embeddings
        with open(os.path.join(self.path, 'ddis-graph-embeddings/entity_ids.del'), 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open(os.path.join(self.path, 'ddis-graph-embeddings/relation_ids.del'), 'r') as ifile:
            self.pred2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2pred = {v: k for k, v in self.pred2id.items()}

        # load the embeddings
        self.entities_graph_embeddings = np.load(os.path.join(self.path, 'ddis-graph-embeddings/entity_embeds.npy'))
        self.predicates_graph_embeddings = np.load(os.path.join(self.path, 'ddis-graph-embeddings/relation_embeds.npy'))
     
        # load the sentence embeddings for predicates and entities
        self.entities_embeddings = np.load(os.path.join(self.path, 'processed/entities_embeddings.npy'))
        self.predicates_embeddings = np.load(os.path.join(self.path, 'processed/predicates_embeddings.npy'))
 
        # images.json for multimedia
        # with open(os.path.join(self.path, 'movienet/images.json'), 'r') as f:
        #    self.images_dict = json.load(f)
        with open(os.path.join(self.path, 'images.pkl'), 'rb') as file: 
            self.images_dict = pickle.load(file)
    
        self.greet_emb = self.sentence_model.encode('Hello, how are you?')

        # for recommendation
        # self.top250 = set(open(os.path.join(self.path, 'imdb-top-250.t')).read().split('\n')) - {''}
        # self.plots = {}
        # with open(os.path.join(self.path, 'plots.csv')) as csvfile:
        #    reader = csv.DictReader(csvfile)
        #    for row in reader:
        #        self.plots[row['qid']] = row['plot']

