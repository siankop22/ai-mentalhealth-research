Literature Review: AI-Driven Mental Health Detection in Low-Resource Languages
1. Introduction

The rapid advancement of natural language processing (NLP) and large language models (LLMs) has catalyzed the development of text-based mental health detection systems. However, most research remains centered on English data, leaving significant gaps for low-resource languages such as Burmese and Zomi. The following review examines major themes, representative studies, and methodological trends in mental health text analysis, multilingual transfer, and ethical deployment.

2. Foundational Research on NLP for Mental Health

Early NLP approaches to mental health detection employed handcrafted features and linear classifiers. Zhang et al. (2022) provide a comprehensive review of 399 NLP studies across ten years, identifying a methodological shift toward deep learning, particularly transformer-based architectures like BERT and RoBERTa. These methods outperform traditional techniques but require large annotated datasets—posing challenges for non-English languages.

Montejo-Ráez et al. (2024) extend this survey, emphasizing the need for cultural and linguistic adaptation in mental health modeling. They highlight the importance of contextual semantics, as linguistic cues for distress vary across cultural groups. This finding supports the inclusion of culturally grounded idioms and expressions—relevant to Burmese and Zomi users—in dataset design.

3. Deep Learning and Transformer Architectures

Recent works, including Hasan et al. (2025), demonstrate that transformer models outperform recurrent networks (LSTM, GRU) in detecting depression and anxiety signals from social media text. These architectures capture long-range dependencies and subtle linguistic markers of emotion. However, they often overfit small datasets and exhibit poor cross-domain generalization.

To mitigate data scarcity, transfer learning and multilingual models have gained attention. Koto et al. (2024) introduce zero-shot sentiment analysis across 34 low-resource languages using lexicon-enhanced multilingual transformers, achieving significant improvements in cross-lingual sentiment prediction. This approach is particularly relevant for Burmese/Zomi contexts, where parallel corpora are minimal.

4. Cross-Lingual and Low-Resource Strategies

For low-resource mental health modeling, three strategies dominate:

Translate-train: Translating English mental health data to the target language, followed by fine-tuning multilingual models like XLM-R.

Translate-test: Training on English and translating test samples back to English at inference time.

Few-shot augmentation: Curating small native-language datasets (200–300 items) to improve model grounding and cultural relevance.

These methods enable effective transfer from high-resource to low-resource domains. Continued pretraining on monolingual Burmese text, as proposed in multilingual NLI studies (Myanmar XNLI, 2024), can further align model representations.

5. Interpretability and Ethical Considerations

Interpretability remains a critical concern in mental health AI. Kim et al. (2025) propose combining LLM-generated summaries with traditional classifiers for transparent depression detection. This hybrid strategy improves both generalization and explainability, helping researchers understand model focus and errors.

Ethically, researchers such as Zhang et al. (2022) and Montejo-Ráez et al. (2024) stress data privacy, informed consent, and cultural sensitivity. Burmese and Zomi datasets must exclude personal identifiers and rely on paraphrased or self-authored examples. Additionally, model deployment should emphasize awareness and triage—not diagnosis—to prevent harm from false positives or negatives.

6. Gaps and Opportunities

Despite significant progress, critical gaps remain:

Lack of open-source, culturally annotated Burmese/Zomi datasets.

Limited exploration of idiomatic and metaphorical expressions in non-English distress detection.

Minimal benchmarking for cross-lingual generalization across South and Southeast Asian languages.

Insufficient ethical documentation and bias auditing for multilingual mental health models.

Future work should focus on integrating human-centered design with technical optimization—balancing predictive accuracy, fairness, and interpretability.

7. Conclusion

The reviewed literature highlights both the promise and challenges of AI-driven mental health detection in low-resource languages. Building upon multilingual transformer architectures (e.g., XLM-R), cross-lingual transfer, and culturally specific dataset design offers a viable path toward equitable and inclusive mental health technology. This study contributes by curating a Burmese/Zomi dataset, implementing transparent classification through a Streamlit prototype, and addressing ethical considerations in deployment.

Key References

Zhang et al. (2022). Natural language processing applied to mental illness detection: a narrative review. npj Digital Medicine.

Montejo-Ráez et al. (2024). A survey on detecting mental disorders with natural language processing. Information Fusion.

Koto et al. (2024). Zero-shot Sentiment Analysis in Low-Resource Languages. EACL.

Kim et al. (2025). Interpretable Depression Detection from Social Media Text Using LLM-Derived Embeddings. arXiv preprint.

Hasan et al. (2025). Advancing Mental Disorder Detection: A Comparative Evaluation of Transformer and LSTM Architectures on Social Media. arXiv preprint.

Myanmar XNLI (2024). Building a Dataset and Exploring Low‑Resource Approaches to Natural Language Inference with Myanmar. arXiv preprint.