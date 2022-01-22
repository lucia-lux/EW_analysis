import os
import pandas as pd
from EW_analysis_utils import nlp_preprocess,nlp_utilities,nlp_utilities_classes

home = 1
if home:
    infiledir = r"C:\Users\Luzia T\UCL\WorkingFromHome\Possible_online_studies\NLP_expressive_writing\analysis\Processed_2"
    writing_dir = r"C:\Users\Luzia T\UCL\WorkingFromHome\Possible_online_studies\NLP_expressive_writing\analysis\writing_data\statements"
else:
    infiledir = r"P:\EW_analysis\analysis\Processed_2"
    writing_dir = r"P:\EW_analysis\analysis\writing\writing_data"

output_dir = os.path.join(os.getcwd(),"output_dir_nlp")
try:
    os.makedirs(output_dir)
except OSError:
    # if directory already exists
    pass

# get input data
writing_df = pd.read_csv(os.path.join(writing_dir, 'writing_df.csv'))
# preprocess
sentences,tokens = nlp_preprocess.preprocess(writing_df.writing)
# add tokens as a column, add word count (raw and tokenized)
writing_df = nlp_utilities.add_col(writing_df,'writing_tokens',tokens)
writing_df = nlp_utilities.add_word_count(writing_df,'writing',tokens)
# check whether word count differs between groups and save results
with open(os.path.join(output_dir,'_'.join(["model_results","kruskal_wallis.txt"])), "w") as f:
    for val in ['word_count_tokenized','word_count_raw']:
        _, pval, posthoc, key = nlp_utilities.kruskal_wallis_func(writing_df,'Group', val)    
        f.write(f"\nP value ({val}) is {pval}.\n")
        if pval<0.05:
            f.write(f"\nConditions differ significantly on {val}.\n")
            f.write(f"\nPosthoc ({val}) is:\n{posthoc}.\n")
            f.write(f"\nThe key is 1 = {key[0]}, 2 = {key[1]}, 3 = {key[2]}\n")
        else:
            f.write(f"\nNo significant between group differences on {val}.\n")

# word frequency (token-based) analysis
wa = nlp_utilities_classes.Word_Freq_Analyzer('day','Group','writing_tokens',output_dir)
for word_pos in [['JJ'],['VB'],['NN']]:
    print(f"Now processing list for: {word_pos}\n")
    top_words_df = wa.func_top_words(writing_df,word_pos, 0)
    top_words_df.to_csv(
                        os.path.join(
                                    output_dir, 
                                    ''.join(
                                            ['most_frequent_',
                                            word_pos.pop(),
                                            '.csv']
                                            )
                                    )
                        )
res_df,clf = wa.tf_idf_scores(writing_df,'writing',1)
wa.plot_confusion_matrix()