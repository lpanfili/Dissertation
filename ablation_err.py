import pandas as pd
import itertools
from sklearn.metrics import f1_score

def remove_columns(df, cols):
	return df.drop(cols, axis=1)

# Yield a list of dataframes, where each element
# is the input dataframe with a single column or
# tuple of columns removed.
# cols is a list of columns or a list of tuples of columns
def remove_columns_sequentially(df, cols):
	for col in cols:
		if type(col) == tuple:
			yield col, remove_columns(df, list(col))
		else:
			yield col, remove_columns(df, [col])
		
# Sequentially ablate each feature (or list of features)
# in to_ablate, train a model, and yield the result
def ablate_features(x, y, to_ablate, train_fn):
	for col, x in remove_columns_sequentially(x, to_ablate):
		f1 = train_fn(x, y)
		print("\tRemoving \"{}\"; f1 is {}".format(col, f1))
		yield {'feature': col, 'f1': f1}

# train_fn is a function that takes x, y and returns f1
# to_ablate can be a list of features or a list of tuples of features
# ablate_worst ablates the feature or tuple of features that results in
# the wost f1 instead of the best f1
def ablate_category(x, y, to_ablate, train_fn, ablate_worst=False):
	remaining = to_ablate
	all_results = []
	while len(remaining) > 1:
		# In each round, remove all of the features we've eliminted so far.
		# If we're ablating tuples of features instead of just features,
		# itertools.chain flattens the list of tuples into one list.
		ablated = remove_columns(x, list(itertools.chain(*(result['removed'] for result in all_results))))
		print("Ablating single feature from {}".format(remaining))
		# Ablate each remaining feature or tuple of features
		results = list(ablate_features(ablated, y, remaining, train_fn))
		# Get the max or min f1 score, depending on ablate_worst
		best = (min if ablate_worst else max)(results, key=lambda result: result['f1'])
		print("Best f1 was {}, removing feature \"{}\"".format(best['f1'], best['feature']))
		# Remove the feature whose exclusion leads to the highest/lowest f1
		remaining = [f for f in remaining if f != best['feature']]
		# Record the result
		all_results.append({
			'removed': best['feature'],
			'best_f1': best['f1'],
			'results': {result['feature']: result['f1'] for result in results}
		})
	return all_results

# Write results obtained from ablate_category to a file. Rows are the list of
# features (or tuples of features) and columns are rounds.
def report_ablation_results(results):
	features = sorted(list(results[0]['results'].keys()))
	results_dict = {'Feature': features}
	for index, round in enumerate(results):
		results_dict['Round {}'.format(index+1)] = [round['results'][feature] if feature in round['results'] else None for feature in features]
	return pd.DataFrame.from_dict(results_dict)