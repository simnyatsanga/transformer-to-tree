import re
import pandas as pd

training_set_file = 'data/geo/geo880_train600.tsv'
validation_set_file = 'data/geo/geo880_dev100.tsv'
test_set_file = 'data/geo/geo880_test280.tsv'

df = pd.read_csv(training_set_file, sep='\t', names=['natural_query', 'logical_query'])
logical_queries = df['logical_query'].tolist()

# Remove the underscore
clean_logical_queries = [re.sub('_', '', x) for x in logical_queries]

# Remove the 'answer' token
clean_logical_queries = [re.sub('answer ', '', x) for x in clean_logical_queries]

# Remove spaces
clean_logical_queries = [re.sub(' ', '', x) for x in clean_logical_queries]

df['logical_query'] = clean_logical_queries

import ipdb; ipdb.set_trace()