from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT = "report.pdf"

def clean(s):
    return (s.replace("\u2014","-").replace("\u2013","-").replace("\u2022","*")
             .replace("\u00b1","+/-").replace("\u2192","->").replace("\u2018","'")
             .replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"'))

class R(FPDF):
    def header(self):
        self.set_font("Helvetica","B",9)
        self.set_text_color(130,130,130)
        self.cell(0,8,clean("RACE Quiz Studio - Project Report"),align="R",
                  new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica","",8)
        self.set_text_color(150,150,150)
        self.cell(0,8,f"Page {self.page_no()}",align="C")

    def h1(self,num,title):
        self.ln(6)
        self.set_font("Helvetica","B",14)
        self.set_text_color(20,20,20)
        self.cell(0,9,clean(f"{num}. {title}"),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.ln(1)

    def h2(self,title):
        self.ln(3)
        self.set_font("Helvetica","B",11)
        self.set_text_color(40,40,40)
        self.cell(0,7,clean(title),new_x=XPos.LMARGIN,new_y=YPos.NEXT)

    def p(self,txt):
        self.set_font("Helvetica","",10.5)
        self.set_text_color(50,50,50)
        self.multi_cell(0,6,clean(txt))
        self.ln(2)

    def li(self,txt):
        self.set_font("Helvetica","",10.5)
        self.set_text_color(50,50,50)
        self.set_x(self.l_margin+4)
        self.multi_cell(self.epw-4,6,clean("* "+txt))

    def tbl(self,headers,rows):
        self.set_font("Helvetica","B",9)
        self.set_fill_color(220,220,220)
        w=self.epw/len(headers)
        for h in headers:
            self.cell(w,7,clean(h),border=1,fill=True,align="C")
        self.ln()
        self.set_font("Helvetica","",9)
        for i,row in enumerate(rows):
            self.set_fill_color(248,248,248) if i%2==0 else self.set_fill_color(255,255,255)
            for c in row:
                self.cell(w,6,clean(str(c)),border=1,fill=True,align="C")
            self.ln()
        self.ln(3)

    def metrics(self,pairs):
        self.set_font("Helvetica","B",13)
        self.set_text_color(30,30,30)
        w=self.epw/len(pairs)
        for _,v in pairs:
            self.cell(w,8,clean(v),align="C")
        self.ln()
        self.set_font("Helvetica","",8.5)
        self.set_text_color(100,100,100)
        for l,_ in pairs:
            self.cell(w,5,clean(l),align="C")
        self.set_text_color(50,50,50)
        self.ln(8)

pdf=R()
pdf.set_margins(20,20,20)
pdf.add_page()

# Title page
pdf.set_font("Helvetica","B",24)
pdf.set_text_color(20,20,20)
pdf.ln(8)
pdf.cell(0,14,"RACE Quiz Studio",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
pdf.set_font("Helvetica","",13)
pdf.set_text_color(80,80,80)
pdf.cell(0,9,"A Reading Comprehension Quiz Generation System",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
pdf.ln(4)
pdf.set_font("Helvetica","",10)
pdf.set_text_color(120,120,120)
pdf.cell(0,7,"CS Final Project  |  FAST NUCES  |  2026",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
pdf.ln(10)

# 1 Abstract
pdf.h1("1","Abstract")
pdf.p("This report describes the design and implementation of RACE Quiz Studio, an end-to-end "
      "reading comprehension quiz generator trained on the RACE dataset [1]. The system takes an "
      "English passage as input and produces a four-option multiple-choice question with graduated "
      "hints. It is built from traditional machine learning components, with a fine-tuned T5-small "
      "model included as a neural comparison baseline. The answer verification component (Model A) "
      "is a soft-vote ensemble of Logistic Regression, Calibrated SVM, and Random Forest classifiers "
      "operating on twelve hand-crafted TF-IDF and lexical features, reaching 36.5% MCQ accuracy "
      "against a 25% random baseline. The distractor generation component (Model B) is a Random "
      "Forest ranker trained on four features including edit distance and sentence-transformer cosine "
      "similarity, reaching 88% accuracy. A Streamlit interface ties both models into a step-by-step "
      "quiz experience with a developer dashboard that shows live metrics and compares traditional "
      "versus neural question generation outputs.")

# 2 Introduction
pdf.h1("2","Introduction & Motivation")
pdf.p("Generating meaningful reading comprehension questions automatically is a genuinely hard problem. "
      "The task requires identifying what facts in a passage are worth asking about, phrasing those "
      "facts as natural questions, and then producing plausible wrong answers that test real "
      "comprehension rather than keyword matching.")
pdf.p("The RACE dataset [1] was constructed from real English examinations for Chinese middle and "
      "high school students. Unlike many reading comprehension datasets, the wrong options in RACE "
      "are written by human teachers to be plausible distractors, not random sentences. This means "
      "models trained on RACE have access to high-quality negative examples, which matters for both "
      "the answer verifier and the distractor ranker.")
pdf.p("The motivation for this project was to understand what traditional ML alone can achieve on "
      "this task before reaching for large neural models. There is a tendency in NLP to jump straight "
      "to transformer fine-tuning, but the compute cost is high and results are not always "
      "interpretable. Building the feature engineering and ensemble from scratch gives a much clearer "
      "picture of which signals actually predict whether an option is correct.")

# 3 Related Work
pdf.h1("3","Related Work")
pdf.p("Du et al. [2] were among the first to frame question generation as a sequence-to-sequence "
      "task, training an attention-based encoder-decoder on SQuAD. Their model learned to identify "
      "answer-worthy spans and generate grammatical questions, though it required large amounts of "
      "labeled pairs. Zhao et al. [3] improved on this with paragraph-level generation using maxout "
      "pointers and gated self-attention, which helps the model stay consistent with broader context.")
pdf.p("On the distractor side, Guo et al. [4] proposed generating distractors from real examination "
      "data using similarity and constraint-based retrieval. The intuition that good distractors come "
      "from the same domain as the correct answer carries through into our Model B design. Jiang and "
      "Lee [5] analyzed click-through data to measure distractor quality and found that edit distance "
      "and semantic similarity between distractor and correct answer are strong quality indicators.")
pdf.p("BERT [6] and T5 [7] are the dominant pre-trained backbones for modern reading comprehension. "
      "We include T5-small fine-tuned on RACE as a neural baseline for question generation quality "
      "comparison. BLEU [8] and ROUGE [9] are standard automatic metrics for generated text, though "
      "both have known limitations when applied to open-ended question generation.")

# 4 Dataset
pdf.h1("4","Dataset Analysis")
pdf.h2("4.1  RACE Overview")
pdf.p("RACE [1] contains 97,687 questions across 27,933 English passages from Chinese middle and "
      "high school exams. Each CSV row has an article, a question, four options (A-D), and the "
      "correct answer label. We used the provided train.csv with 87,866 articles after deduplication.")
pdf.h2("4.2  Key Statistics")
pdf.tbl(
    ["Metric","Value"],
    [["Total articles (train)","87,866"],
     ["Questions per article (avg)","~1.1"],
     ["Avg article length (tokens)","~321"],
     ["Avg question length (tokens)","~11"],
     ["Answer label distribution","~25% each (A/B/C/D)"],
     ["Avg option length (tokens)","~5.2"],
     ["Training rows (after 1:1 balance)","149,372"],
     ["Test rows (natural split)","52,720"]])
pdf.h2("4.3  Observations")
pdf.li("Answer labels are roughly balanced across A, B, C, D with slight bias toward A in the middle school subset.")
pdf.li("About 67% of questions are Wh-questions. The remainder are cloze-style fill-in-the-blank.")
pdf.li("Correct answers appear verbatim in the article in approximately 58% of cases, making exact-match a useful feature.")
pdf.li("Distractor options rarely contain the correct answer string, confirming exact-match is a strong negative signal.")
pdf.li("Article length follows a roughly log-normal distribution. High school articles exceed 600 tokens.")
pdf.ln(3)

# 5 Model A
pdf.h1("5","Model A: Answer Verification & Question Generation")
pdf.h2("5.1  Problem Framing")
pdf.p("Model A has two jobs: (1) generate a question for an extracted answer candidate, and (2) score "
      "four answer options to identify the correct one. Job (1) is handled by a rule-based template "
      "system followed by an RF question ranker. Job (2) is binary classification: given a "
      "(article, question, option) triple, predict whether the option is the correct answer.")
pdf.h2("5.2  Answer Candidate Extraction")
pdf.p("The system identifies answer candidates using regular expressions across six pattern types: "
      "dates, acronyms, possessive names, multi-word proper nouns, definition phrases (known as / "
      "called / referred to as), and location phrases (in/at/near [place]). Each candidate gets "
      "a heuristic score based on type and sentence position. The top eight are passed to the ranker.")
pdf.h2("5.3  Feature Engineering (12 features)")
pdf.tbl(
    ["#","Feature","Description"],
    [["0","tfidf_cosine_sim","TF-IDF cosine sim, article vs (question+option)"],
     ["1","unigram_overlap_ratio","Fraction of (Q union O) tokens found in article"],
     ["2","idf_weighted_score","Mean IDF weight of option tokens shared with article"],
     ["3","best_sentence_overlap","Max unigram overlap over any single article sentence"],
     ["4","bigram_overlap_ratio","Bigram version of feature 1"],
     ["5","exact_match_flag","1 if option appears verbatim in article"],
     ["6","option_token_length","Token count of option text"],
     ["7","option_char_length","Character count of option text"],
     ["8","first_position","Normalised first-occurrence position in article"],
     ["9","is_number","1 if option is numeric or date-like"],
     ["10","kmeans_cluster_label","K-Means (k=2) hard cluster assignment"],
     ["11","kmeans_centroid_dist","Distance to nearest K-Means centroid"]])
pdf.p("Feature 3 (best_sentence_overlap) is the single most discriminative feature. The correct "
      "answer typically aligns strongly with one specific sentence while distractors spread their "
      "overlap across many sentences. Features 10 and 11 come from an unsupervised step: "
      "TruncatedSVD(64) reduces option TF-IDF vectors, then K-Means(k=2) assigns cluster labels. "
      "These add a small but consistent improvement to ensemble F1.")
pdf.h2("5.4  Training Procedure")
pdf.p("Training data: 87,866 RACE articles exploded into 4 option rows each, then 1:1 downsampled "
      "to 149,372 balanced rows. Test set: natural 75/25 split giving 52,720 rows (no resampling). "
      "Using class_weight='balanced' inside sklearn caused Calibrated SVM to predict all-negative. "
      "Manual 1:1 sampling before fitting produced better-calibrated probabilities for all models.")
pdf.li("Logistic Regression: C=1.0, max_iter=2000")
pdf.li("Calibrated SVM: LinearSVC(C=1.0) in CalibratedClassifierCV(method='isotonic', cv=3)")
pdf.li("Naive Bayes: GaussianNB - trained and reported, excluded from ensemble vote")
pdf.li("Random Forest: 300 trees, max_depth=15, min_samples_leaf=5, max_features='sqrt'")
pdf.li("TF-IDF vectorizer: max_features=20,000, sublinear_tf=True")
pdf.ln(3)
pdf.h2("5.5  RF Question Ranker")
pdf.p("A separate RF ranker is trained on 38 features per (question, answer candidate) pair, "
      "including question length, wh-word identity, template type (suffix_wh / definition / "
      "where_wh / cloze / generic), source sentence position, answer kind, and five Model A "
      "verifier scores. Trained on RACE gold (question, answer) pairs. Result: ROC-AUC 0.972, "
      "top-1 match rate 66.9% on held-out set.")
pdf.h2("5.6  Neural Baseline: T5-small")
pdf.p("For comparison, T5-small was fine-tuned on 20,000 RACE Wh-questions for 5 epochs "
      "(batch 16, lr=3e-4). Input: 'generate question: context: <article> answer: <answer>'. "
      "The model generates fluent questions but occasionally produces generic outputs that could "
      "fit many articles. Template+ranker questions are less fluent but more specific on average.")
pdf.h2("5.7  Results")
pdf.metrics([("MCQ Accuracy","36.5%"),("Binary Accuracy","57.9%"),
             ("Binary F1","0.376"),("Cross-Val F1","0.526 +/- 0.002"),("Training Rows","149,372")])
pdf.tbl(
    ["Classifier","In Vote","Accuracy","Precision","Recall","F1"],
    [["Logistic Regression","Yes","0.5689","0.2907","0.5031","0.3685"],
     ["Calibrated SVM","Yes","0.5746","0.2917","0.4913","0.3661"],
     ["Naive Bayes","No","0.6579","0.3121","0.3058","0.3089"],
     ["Random Forest","Yes","0.5864","0.3043","0.5086","0.3808"],
     ["Ensemble (LR+SVM+RF)","--","0.5788","--","--","0.3762"]])
pdf.p("Confusion matrix on 52,720 test rows: TP=6,698  FP=15,726  FN=6,482  TN=23,814. "
      "MCQ accuracy of 36.5% is 11.5 points above the 25% random baseline. Naive Bayes is "
      "excluded from the vote because its recall (~30%) is much lower than LR/RF (~50%), which "
      "suppresses the correct option's soft-vote score and hurts MCQ ranking.")

# 6 Model B
pdf.h1("6","Model B: Distractor Generation & Hint System")
pdf.h2("6.1  Overview")
pdf.p("Model B supplies three plausible wrong answer options and three graduated hints. "
      "Candidates come from an offline-built index of 87,866 RACE articles. At inference time "
      "the retriever finds options from other questions sharing the same answer category "
      "(date, person, acronym, proper noun, etc.), filters by compatibility rules to avoid "
      "semantic leakage, then ranks them with a trained Random Forest.")
pdf.h2("6.2  Distractor Ranker Features (4 features)")
pdf.tbl(
    ["Feature","Description"],
    [["Edit distance","Levenshtein distance between correct answer and candidate"],
     ["Length difference","Absolute character length difference"],
     ["In-article flag","1 if candidate text appears in the source article"],
     ["Cosine similarity","Sentence-transformer (all-MiniLM-L6-v2) cosine sim to correct answer"]])
pdf.h2("6.3  Training")
pdf.p("Training data: 3,000 sampled RACE rows. For each row, the three non-correct options are "
      "labeled positive (good distractors, label=1) and a set of WordNet generic terms (object, "
      "thing, unknown, item) are labeled negative (bad distractors, label=0). Random Forest "
      "(150 trees, max_depth=12) was compared against a Logistic Regression baseline.")
pdf.h2("6.4  Results")
pdf.metrics([("RF Accuracy","88.0%"),("RF Precision","85.0%"),
             ("RF Recall","82.0%"),("RF F1","0.83")])
pdf.h2("6.5  Hint Generation")
pdf.p("Three graduated hints are generated entirely by rule, with no ML at inference time. "
      "This was intentional: rule-based hints are fast, deterministic, and easy to debug.")
pdf.li("Hint 1 (General): First sentence in the article that does not mention the answer.")
pdf.li("Hint 2 (Intermediate): Sentence immediately before the answer sentence.")
pdf.li("Hint 3 (Specific): Answer sentence with the answer replaced by '[BLANK]'.")
pdf.ln(3)

# 7 UI
pdf.h1("7","User Interface")
pdf.p("The front end is a Streamlit app (ui/app.py) with four screens managed through session state. "
      "A background daemon thread pre-loads the pipeline on first page visit so the UI is immediately "
      "responsive and the user never sees a cold-start delay. The pipeline singleton is cached via "
      "@st.cache_resource, so subsequent calls return instantly.")
pdf.li("Screen 1 - Article Input: paste text, load a built-in sample, or pick from 500 RACE rows.")
pdf.li("Screen 2 - Quiz View: generated question with four radio options. After answering, shows "
       "verifier probability scores for all four options as a progress bar table.")
pdf.li("Screen 3 - Hint Panel: progressive hint reveal (Hint 1 -> 2 -> 3 -> Reveal Answer).")
pdf.li("Screen 4 - Developer Dashboard: latency tile, confusion matrix, per-classifier table, "
       "T5 vs template question comparison, and session log CSV export.")
pdf.ln(3)

# 8 Evaluation
pdf.h1("8","Evaluation & Discussion")
pdf.h2("8.1  Answer Verification")
pdf.p("The 36.5% MCQ accuracy is above random (25%) but well below human performance on RACE (~94%). "
      "This is expected: twelve lexical features cannot handle inference questions, which make up a "
      "large fraction of RACE. The ensemble mainly catches factual extraction questions where the "
      "correct answer appears verbatim or near-verbatim in the article.")
pdf.p("The 5-fold cross-validation F1 of 0.526 is higher than the held-out test F1 of 0.376. "
      "The gap comes from distribution shift: cross-val runs on the balanced training data while "
      "the test set uses the natural 25/75 class ratio. MCQ accuracy better reflects real task "
      "performance because it asks whether the highest-scored option is correct, not whether the "
      "binary threshold is tuned correctly.")
pdf.h2("8.2  Question Generation Quality")
pdf.p("No automated BLEU/ROUGE evaluation was run against gold RACE questions. Manual review of "
      "50 generated questions showed suffix_wh templates produce grammatical questions ~72% of "
      "the time. Cloze questions are always grammatical but feel mechanical. T5 outputs are more "
      "fluent but occasionally hallucinate names not present in the passage.")
pdf.h2("8.3  Distractor Quality")
pdf.p("The 88% distractor ranker accuracy measures whether a candidate beats a generic WordNet term, "
      "not whether it would fool a human. Qualitatively, distractors retrieved from the same category "
      "are usually plausible. Mismatches slip through when the article is short and the retrieval "
      "pool is small, causing the system to fall back to passage candidates from different categories.")

# 9 Limitations
pdf.h1("9","Limitations & Future Work")
pdf.li("MCQ accuracy is bottlenecked by features. Adding NER type matching, dependency parse "
       "features, or a sentence embedder would meaningfully improve ranking.")
pdf.li("The distractor ranker trained on only 3,000 articles. Full dataset training would improve quality.")
pdf.li("Inference questions requiring cross-sentence reasoning are essentially unsolvable with "
       "lexical features. A BERT entailment model would handle these cases.")
pdf.li("No quantitative BLEU/ROUGE evaluation of question quality. This should be added.")
pdf.li("Hint generation is rule-based and can reveal too much or too little depending on article structure.")
pdf.li("The distractor index covers only the training split. A broader corpus would increase option diversity.")
pdf.ln(3)

# 10 Conclusion
pdf.h1("10","Conclusion")
pdf.p("We built a complete reading comprehension quiz generation pipeline using traditional ML, "
      "with a neural baseline for comparison. The system extracts answer candidates with regex, "
      "generates template questions, ranks them with a trained RF ranker, retrieves and ranks "
      "distractors from the RACE corpus, and produces three graduated hints. The answer verifier "
      "ensemble reaches 36.5% MCQ accuracy (11.5 points above random) on the RACE test set. "
      "The distractor ranker reaches 88% accuracy distinguishing good distractors from generic terms.")
pdf.p("The biggest lesson from this project is that feature engineering matters. The "
      "best_sentence_overlap feature alone accounts for a large share of the ensemble's discriminative "
      "power. Traditional ML with well-designed features remains a reasonable baseline for this kind "
      "of task, and the interpretability is a genuine advantage over neural approaches.")

# 11 References
pdf.h1("11","References")
refs=[
    "[1] Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. EMNLP 2017. https://aclanthology.org/D17-1082",
    "[2] Du, X., Shao, J., & Cardie, C. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. ACL 2017. https://aclanthology.org/P17-1123",
    "[3] Zhao, Y., Ni, X., Ding, Y., & Ke, Q. (2018). Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks. EMNLP 2018. https://aclanthology.org/D18-1424",
    "[4] Guo, Q., Zhu, X., Chao, W., & Ye, Z. (2016). Generating Distractors for Reading Comprehension Questions from Real Examinations. AAAI 2016. https://ojs.aaai.org/index.php/AAAI/article/view/10395",
    "[5] Jiang, Y., & Lee, H.-Y. (2017). What Click-Through Data Can Tell Us About Distractor Quality. BEA Workshop 2017. https://aclanthology.org/W17-5019",
    "[6] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. https://aclanthology.org/N19-1423",
    "[7] Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR 21(140). https://jmlr.org/papers/v21/20-1166.html",
    "[8] Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002. https://aclanthology.org/P02-1040",
    "[9] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop 2004. https://aclanthology.org/W04-1013",
]
pdf.set_font("Helvetica","",9.5)
pdf.set_text_color(50,50,50)
for r in refs:
    pdf.multi_cell(0,6,clean(r))
    pdf.ln(2)

pdf.output(OUT)
print(f"Saved: {OUT}")
