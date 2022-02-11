import pandas as pd

def load(path):
	df = None
	'''YOUR CODE HERE'''
	df = pd.read_csv(path, encoding='latin-1')
	'''END'''
	return df

def prior(df):
	ham_prior = 0
	spam_prior = 0
	'''YOUR CODE HERE'''
	#calculate the prior probability of each class
	ham_prior = df['label'].value_counts()[1]/df.shape[0]
	spam_prior = df['label'].value_counts()[0]/df.shape[0]
	'''END'''
	return ham_prior, spam_prior

def likelihood(df):
	ham_like_dict = {}
	spam_like_dict = {}
	'''YOUR CODE HERE'''
	#create a dictionary of words and their probability for each class
	#ham_like_dict = {'word': probability}
	ham_mail = df.loc[df['label'] == 'ham', 'text'].values
	for i in range(len(ham_mail)):
		for word in ham_mail[i].split():
			if word in ham_like_dict:
				ham_like_dict[word] += 1
			else:
				ham_like_dict[word] = 1
	ham_like_dict = {k: v/df['label'].value_counts()[1]
                  for k, v in ham_like_dict.items()}

	#spam_like_dict = {'word': probability}
	spam_mail = df.loc[df['label'] == 'spam', 'text'].values
	for i in range(len(spam_mail)):
		for word in spam_mail[i].split():
			if word in spam_like_dict:
				spam_like_dict[word] += 1
			else:
				spam_like_dict[word] = 1
	spam_like_dict = {k: v/df['label'].value_counts()[0] for k, v in spam_like_dict.items()}
	'''END'''

	return ham_like_dict, spam_like_dict

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
	'''
	prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
	'''
	#ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
	ham_spam_decision = None

	'''YOUR CODE HERE'''
	#ham_posterior = posterior probability that the email is normal/ham
	ham_posterior = None

	#spam_posterior = posterior probability that the email is spam
	spam_posterior = None

	'''END'''
	return ham_spam_decision


def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
	'''
	Calls "predict"
	'''
	hh = 0  # true negatives, truth = ham, predicted = ham
	hs = 0  # false positives, truth = ham, pred = spam
	sh = 0  # false negatives, truth = spam, pred = ham
	ss = 0  # true positives, truth = spam, pred = spam
	num_rows = df.shape[0]
	for i in range(num_rows):
		roi = df.iloc[i, :]
		roi_text = roi.text
		roi_label = roi.label_num
		guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
		if roi_label == 0 and guess == 0:
			hh += 1
		elif roi_label == 0 and guess == 1:
			hs += 1
		elif roi_label == 1 and guess == 0:
			sh += 1
		elif roi_label == 1 and guess == 1:
			ss += 1

	acc = (ss + hh)/(ss+hh+sh+hs)
	precision = (ss)/(ss + hs)
	recall = (ss)/(ss + sh)
	return acc, precision, recall


if __name__ == "__main__":
	'''YOUR CODE HERE'''
	#this cell is for your own testing of the functions above
	df = load('TRAIN_balanced_ham_spam.csv')
	ham_prior, spam_prior = prior(df)
	ham_like_dict, spam_like_dict = likelihood(df)
