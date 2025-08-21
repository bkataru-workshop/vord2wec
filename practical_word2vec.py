import os
import pickle
import re
import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from mltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
