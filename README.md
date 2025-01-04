# Legal AI Risk Management Chatbot ⚖️🚨

This repository contains the code and resources for the **Legal AI Risk Management Chatbot**, a project developed as part of the AIE4 Midterm Assignment. The chatbot is designed to help users understand and manage risks associated with AI by leveraging Retrieval-Augmented Generation (RAG) and fine-tuned embedding models.

---

## Key Features ✨

- **RAG Pipeline**: A robust Retrieval-Augmented Generation system to answer user queries based on AI risk management frameworks.
- **Fine-Tuned Embeddings**: Custom fine-tuned embeddings using the `snowflake-arctic-embed-m` model for improved performance on domain-specific tasks.
- **Evaluation Framework**: Comprehensive evaluation using the RAGAS framework to measure faithfulness, answer relevancy, context precision, and context recall.
- **Deployment**: The chatbot is deployed on Hugging Face Spaces for easy access and testing.

---

## Repository Structure 📂

```
Legal-AI-Risk-Management/
├── data/                   # Contains datasets and evaluation results
├── notebooks/              # Jupyter notebooks for RAG setup, evaluation, and fine-tuning
│   ├── rag_and_ragas_pipelines.ipynb       # RAG setup and evaluation
│   └── finetuning_embeddings_model.ipynb   # Fine-tuning pipeline
├── myutils/                # Custom utility modules for refactored code
├── results/                # Output files, including visualizations and performance metrics
├── README.md               # This file
└── requirements.txt        # List of dependencies
```

---

## Getting Started 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Famz21/Legal-AI-Risk-Management.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the Notebooks**:
   - **Main Notebook**: [rag_and_ragas_pipelines.ipynb](rag_and_ragas_pipelines.ipynb) contains the RAG setup and evaluation.
   - **Fine-Tuning Notebook**: [finetuning_embeddings_model.ipynb](finetuning_embeddings_model.ipynb) details the fine-tuning process for the embedding model.

4. **Run the Application**:
   - The chatbot is deployed on Hugging Face Spaces. Access it here: [Hugging Face App](https://huggingface.co/spaces/Vira21/Legal_AI_Risk_Management).

---

## Key Deliverables 📦

1. **Fine-Tuned Embedding Model**:
   - Model: `Vira21/finetuned_arctic`
   - Link: [Fine-Tuned Model on Hugging Face](https://huggingface.co/Vira21/finetuned_arctic)

2. **Hugging Face Deployment**:
   - App: [Legal AI Risk Management Chatbot](https://huggingface.co/spaces/Vira21/Legal_AI_Risk_Management)

3. **Loom Video Demo**:
   - Watch the prototype in action: [Loom Video]()

---

## Evaluation Results 📊

The RAG pipeline was evaluated using the RAGAS framework. Below are the key metrics for different configurations:

| Configuration                     | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-----------------------------------|--------------|------------------|-------------------|----------------|
| OpenAI Embeddings (Baseline)      | 0.92         | 0.89             | 0.91              | 0.90           |
| OpenAI Embeddings (Semantic Chunking) | 0.93      | 0.90             | 0.92              | 0.91           |
| Snowflake-Arctic-Embed-M (Baseline) | 0.75        | 0.72             | 0.70              | 0.68           |
| Finetuned Arctic Embeddings       | 0.93         | 0.91             | 0.92              | 0.91           |

**Key Takeaways**:
- Fine-tuning significantly improved the performance of the `snowflake-arctic-embed-m` model, making it competitive with OpenAI embeddings.
- Semantic chunking provided marginal improvements in retrieval and generation metrics.

---

## Recommendations 💡

For internal stakeholder testing, we recommend using the **fine-tuned `snowflake-arctic-embed-m` model** with **semantic chunking**. This configuration balances performance, cost-effectiveness, and domain-specific accuracy.

---

## Future Work 🔮

1. **Incorporate Additional Documents** 📜:
   - Add the 270-day update on the 2023 Executive Order on Safe, Secure, and Trustworthy AI.
   - Implement a persistent vector database to handle document updates efficiently.

2. **Enhance Retrieval Architecture** 🛡️:
   - Use metadata to manage document versions and improve retrieval relevance.

3. **Expand Evaluation** ⚖️:
   - Conduct user testing with internal stakeholders to gather feedback and further refine the system.

---

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact 📧

For questions or feedback, please reach out to [your email or contact information].

---

Happy exploring! 🚀