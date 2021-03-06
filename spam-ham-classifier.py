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
	#alpha = 1
	#ham_alpha_count = 0
	#spam_alpha_count = 0
	ham_word_count = {}
	spam_word_count = {}
	
	#create a list of all text from ham emails
	ham_mail = df.loc[df['label'] == 'ham', 'text'].values
	
	#iterate through each word in each email, counting the number of emails that contain a given word
	for mail in ham_mail:
		words = []
		for word in mail.split():
			if word not in words:
				words.append(word)
		for word in words:
			if word not in ham_word_count:
				ham_word_count[word] = 1 #+ alpha
				#ham_alpha_count += 1
			else:
				ham_word_count[word] += 1
			
			# if word not in spam_word_count:
			# 	spam_word_count[word] = alpha
			# 	spam_alpha_count += 1

	#create a list of all text from spam emails
	spam_mail = df.loc[df['label'] == 'spam', 'text'].values

	#iterate through each word in each email, counting the number of emails that contain a given word
	for mail in spam_mail:
		words = []
		for word in mail.split():
			if word not in words:
				words.append(word)
		for word in words:
			if word not in spam_word_count:
				spam_word_count[word] = 1 #+ alpha
				# spam_alpha_count += 1
			else:
				spam_word_count[word] += 1

			# if word not in ham_word_count:
			# 	ham_word_count[word] = alpha
			# 	ham_alpha_count += 1
		
	#calculate the likelihood of each word in the 'ham' class
	ham_like_dict = {k: v/(df['label'].value_counts()[1])
                  for k, v in ham_word_count.items()}
	# ham_like_dict = {k: v/(df['label'].value_counts()[1] + ham_alpha_count)
    #               for k, v in ham_word_count.items()}


	#calculate the likelihood of each word in the 'spam' class
	spam_like_dict = {k: v/(df['label'].value_counts()[0])
				  for k, v in spam_word_count.items()}
	# spam_like_dict = {k: v/(df['label'].value_counts()[0] + spam_alpha_count)
    #                for k, v in spam_word_count.items()}
	'''END'''

	return ham_like_dict, spam_like_dict

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
	'''
	prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
	'''
	#ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
	ham_spam_decision = None

	'''YOUR CODE HERE'''
	alpha = 1.4
	p_words_ham = 1.0
	p_words_spam = 1.0

	for i in text.split():
		if i in ham_like_dict:
			p_words_ham *= (ham_like_dict[i] + alpha)

	for i in text.split():
		if i in spam_like_dict:
			p_words_spam *= (spam_like_dict[i] + alpha)
	
	#ham_posterior = posterior probability that the email is normal/ham
	ham_posterior = (ham_prior * p_words_ham)/(ham_prior * p_words_ham + spam_prior * p_words_spam)
	

	#spam_posterior = posterior probability that the email is spam
	spam_posterior = (spam_prior * p_words_spam)/(ham_prior * p_words_ham + spam_prior * p_words_spam)

	if ham_posterior > spam_posterior:
		ham_spam_decision = 0
	else:
		ham_spam_decision = 1

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
	df_test = load('TEST_balanced_ham_spam.csv')
	ham_prior, spam_prior = prior(df)
	ham_like_dict, spam_like_dict = likelihood(df)
	# spam = df_test.loc[df_test['label'] == 'spam',
    #                      'text'].values
	# spam_test = spam[random.randint(0, len(spam)-1)]
	# ham = df_test.loc[df_test['label'] == 'ham', 'text'].values[0]
	# ham_test = ham[random.randint(0, len(ham)-1)]
	# print(predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, spam_test))
	# print(predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, ham_test))
	print(metrics(ham_prior, spam_prior, ham_like_dict, spam_like_dict, df_test))
