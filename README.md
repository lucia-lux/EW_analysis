# EW_analysis
Preprocessing and analysis of data from the Expressive Writing study.

Work in progress, but this repository will eventually contain all the scripts used for analysing the data from the Expressive writing study.
More details on this study are available here:
https://osf.io/7f5zu/

Some changes to the analysis approach described:
(1) Switched from linear mixed effects models to GEE - better suited given the fact that we're interested in population-averaged responses, rather than differences between individuals per se; better given the distributions of the response variable. Used Tweedie to model mean response structure, log link, first order autoregressive covariance structure (unless completely inappropriate, in which case we used Independence). AR1 may not be the 'true' covariance structure, but coefficient estiamtes are robust to mis-specification (but efficiency suffers). Standard errors may be affected, but default in statsmodels implementation is to use robust estimation (Huber White?), so should be ok.]
(2) Exploratory: Added moderator (chronicity of stressful event, as indicated by reporting of the same event at each assessment time point).
(3) Exploratory: NLP - not going to do exactly what we stated as text corpus too small and it's proven difficult to get access to similar type content that could be used to train a model. Still using NLP to do manipulation check/some topic modeling to get an idea of writing content. Not entirely decided on what else yet, work in progress.
(4) LDI: Scores were way lower than expected. No specific patterns/improvement, which may be an indication that the MST is a task that is best suited to being used in the laboratory, not online as was done here.

Data extraction files relatively complete (although could be more elegant).
Main analysis file (GEE) - done, but could do with cleaning up/making it more elegant.
NLP - work in progress.
