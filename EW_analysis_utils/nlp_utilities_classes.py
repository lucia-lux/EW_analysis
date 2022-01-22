import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import FreqDist
from nltk.corpus import wordnet
from sklearn import feature_extraction, model_selection, pipeline, metrics
from sklearn.svm import LinearSVC
from nltk import WordNetLemmatizer
from EW_analysis_utils.nlp_utilities import flatten_list

class Word_Freq_Analyzer:
    """
    Class for word frequency based analysis.
    """
    def __init__(self, day_col_name, group_col_name, token_col_name,out_dir):
        self.day_col = day_col_name
        self.group_col = group_col_name
        self.token_col = token_col_name
        self.out_dir = out_dir

    def get_pos_tag(self, tag):
        """
        Get wordnet pos tag.

        Parameters
        ----------
        tag:    str
        POS tag (from nltk pos_tag output tuple)

        Returns
        -------
        wordnet pos tag
        can be passed to nltk lemmatizer.
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def func_lemmatize(self, word_list):
        """
        Lemmatize word list.

        Parameters
        ----------
        word_list:  list
            words to process
        
        Returns
        -------
        Lemmatized word list.

        """
        lemmatizer = WordNetLemmatizer()
        word_list = nltk.pos_tag(word_list)
        words_lemmatized = [lemmatizer.lemmatize(word,self.get_pos_tag(tag))
                            for (word,tag) in word_list]
        return words_lemmatized
    
    def get_top_words(self, num_day, group_name, in_df, **pos_tag_type):
        """ 
        Get an ordered list of words in document.

        Parameters
        ----------
        num_day:    int
            Day of writing (1, 2, 3 or 4)
        group_name: str
            The group to process (EW, EWRE or CTR)
        in_df:  pd DataFrame
            input dataframe containing rel data
        **pos_tag_type: list of str
            If processing only nouns/verbs/adjectives
            pass tag to function using kwargs.
            For adjectives, use:
            'JJ'
            For verbs, use:
            'VB'
            For nouns, use:
            'NN'
            If all of the above, pass list:
            ['NN','JJ','VB']
            as kwarg.

        Returns
        -------
        list of words, list of vals
        words = words ordered from most frequent to rare
        vals = corresponding frequency
        """
        token_list = [
                    item for sublist in
                    [*in_df.loc[
                    (in_df[self.group_col] == group_name) &
                    (in_df[self.day_col] == num_day),
                    self.token_col]]
                    for item in sublist
                    ]

    
        if pos_tag_type:
            selected_list = []
            for tag_type in pos_tag_type.values():
                w_list = [
                        word for (word,pos) 
                        in nltk.pos_tag(token_list)
                        for tag in tag_type
                        if pos[:2] == tag
                        ]
                selected_list.extend(w_list)
        else:
            selected_list = token_list
        freqs = FreqDist(selected_list)
        common_tups = freqs.most_common()
        self.common_words = list(zip(*common_tups))[0]
        self.common_vals = list(zip(*common_tups))[1]
        return self.common_words, self.common_vals
    
    def func_top_words(self, in_df, pos_tags, visualize):
        """
        Put top 50 words in dataframe,
        with option to visualize using barplots.

        Parameters
        ----------
        in_df:  pd DataFrame
        input dataframe
        pos_tags:   list
            list of pos tags to use
            can be VB, JJ, NN or
            any combination (or all) of these
        visualize:  int
        1 if visualization is needed
        0 otherwise

        Returns
        -------
        Datframe of 50 top words
        and their frequencies for 
        all days and conditions.

        """

        top_50_words = []
        top_50_vals = []
        condition = []
        days = []

        for group_name in in_df[self.group_col].unique():
            for day in in_df[self.day_col].unique():
                words,vals= self.get_top_words(day, group_name, in_df, pos_tags = pos_tags)
                top_50_words.append(list(words[:50]))
                top_50_vals.append(list(vals[:50]))
                condition.append(np.repeat(group_name,50))
                days.append(np.repeat(day,50))

        data = {
            'words': flatten_list(top_50_words),
            'vals': flatten_list(top_50_vals),
            'day': flatten_list(days),
            'group': flatten_list(condition)
            }
        self.most_common_words_df = pd.DataFrame(data)

        if visualize == 1:
            for num_day in self.most_common_words_df.day.unique():
                fig,axes = plt.subplots(3,1,figsize = (30,15),sharey = True)
                for i, group_name in enumerate(self.most_common_words_df.group.unique()):
                    data = self.most_common_words_df.loc[
                                                    (self.most_common_words_df.day==num_day) &
                                                    (self.most_common_words_df.group == group_name),
                                                    ['words','vals']
                                                    ]
                    sns.barplot(ax=axes[i], x=data.words, y=data.vals)
                    axes[i].set_title(f'Condition: {group_name}, Day: {num_day}')
                fig.savefig("most_common_words.png")

        return self.most_common_words_df
    
    def print_vect_feats(self, vectorizer, clf):
        """
        Prints features with the highest coefficient values,
        per class

        """
        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(clf.classes_):
            features_sorted = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" % (class_label,
            " ".join(feature_names[j] for j in features_sorted)))
    
    def save_vect_feats(self, vectorizer, clf):
        """
        saves features with the highest coefficient values,
        per class (maybe...)

        """
        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(clf.classes_):
            features_sorted = np.argsort(clf.coef_[i])[-50:]
            data = {'features_sorted':features_sorted,
                    '_'.join(['feature_names',class_label]): [feature_names[j] for j in features_sorted]}
            out_df = pd.DataFrame(data)
            out_df.to_csv(os.path.join(self.out_dir,'_'.join([class_label,'differentiating_features.csv'])),index = False)
    
    def get_coefficients(self, clf, feature_names, num_feats):
        """
            Gets top/worst num_feats features used to classify
            statements.
            Returns bar plots 
        """
        for i, class_label in enumerate(clf.classes_):
            coef = clf.coef_[i].ravel()
            top_positive_coefficients = np.argsort(coef)[-num_feats:]
            top_negative_coefficients = np.argsort(coef)[:num_feats]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            # create plot
            plt.figure(figsize=(15, 5))
            plt.title(f"Best/worst predictors, condition:{class_label}")
            colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
            plt.bar(np.arange(2 * num_feats), coef[top_coefficients], color=colors)
            feature_names = np.array(feature_names)
            plt.xticks(np.arange(1, 1 + 2 * num_feats), feature_names[top_coefficients], rotation=60, ha='right')
            #plt.show()
            plt.savefig(os.path.join(self.out_dir, '_'.join([class_label,'top_features.png'])))
      #  return top_coefficients

    def tf_idf_scores(self, in_df, writing_col,*cleaned):
        """
        Classify statements using Linear SVC
        and print top 10 distinguishing features

        in_df:  pd DataFrame
            input dataframe
        
        writing_col:    str
            name of column containing written statements
        
        Returns
        -------
        Dataframe containing predictions
        and actual class labels
        for holdout test set
        """
        if cleaned:
            in_df['writing_cleaned'] = in_df[self.token_col].apply(lambda x: ' '.join(x))
            writing_col = 'writing_cleaned'
        
        self.wrdf_train,self.wrdf_test = model_selection.train_test_split(in_df.loc[:,[writing_col, self.group_col]],
                                                                test_size = 0.3,random_state = 35,
                                                                stratify = in_df[self.group_col])
        y_train = self.wrdf_train[self.group_col]
        y_test = self.wrdf_test[self.group_col]
        vectorizer_tf_idf = feature_extraction.text.TfidfVectorizer(strip_accents = 'ascii',
                            stop_words = ['etc','u','especially'],
                            sublinear_tf = True)
        clf = LinearSVC(C=1.0, class_weight="balanced")
        tf_idf = pipeline.Pipeline([('tfidf', vectorizer_tf_idf),("classifier", clf)])
        tf_idf.fit(self.wrdf_train[writing_col], y_train)
        predicted = tf_idf.predict(self.wrdf_test[writing_col])

        self.res_df = pd.DataFrame({'actual': y_test.values, 'predicted': predicted})
        self.print_vect_feats(vectorizer_tf_idf,clf)
        self.save_vect_feats(vectorizer_tf_idf,clf)
        self.get_coefficients(clf,vectorizer_tf_idf.get_feature_names(),20)

        return self.res_df


    
    def tf_idf_features(self, in_df, writing_col,*cleaned):
        """
        get TF-IDF features
        
        Parameters
        ----------
        in_df: pd DataFrame
            input dataframe
        writing_col:    str
            name of column containing
            written statements
        *cleaned:
            if supplied, methods will
            be applied to cleaned text data
        
        Returns
        -------
        dataframe containing features
        and TF-IDF score

        """
        
                            
        if cleaned:
            corpus = in_df[self.token_col].apply(lambda x: ' '.join(x))
        else:
            corpus = in_df[writing_col]
        vectorizer_tf_idf = feature_extraction.text.TfidfVectorizer(sublinear_tf = True)
        features = vectorizer_tf_idf.fit_transform(corpus)
        feats_df = pd.DataFrame(features[0].T.todense(), index=vectorizer_tf_idf.get_feature_names(), columns=["TF-IDF"])
        self.feats_df = feats_df.sort_values('TF-IDF', ascending=False)

        return self.feats_df
    
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix
        for classifier results.
        """
        # Plot the confusion matrix.
        y_test = self.res_df['actual']
        classes = np.unique(y_test)
        #y_test_array = pd.get_dummies(y_test, drop_first=False).values    
        cm = metrics.confusion_matrix(y_test, self.res_df['predicted'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels= classes, yticklabels=classes, title="Confusion matrix")
        plt.yticks(rotation=0)
        plt.savefig("confusion_matrix.png")