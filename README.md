# Legal AI Risk Management Chatbot ⚖️🚨

This repository contains the code and resources for the **Legal AI Risk Management Chatbot**, a project designed to help users understand and manage risks associated with AI by leveraging Retrieval-Augmented Generation (RAG) and fine-tuned embedding models. The chatbot provides insights into AI risk frameworks, ethical considerations, and best practices, enabling teams to integrate AI responsibly into their workflows.

---

### **Overview**

The **AI Risk Chatbot** was developed to educate internal teams on understanding, measuring, and managing AI risks. It provides actionable insights into risk frameworks and best practices, enabling stakeholders to navigate AI's potential responsibly. Rolled out to **50 teams** over a month, this initiative empowers employees to integrate AI thoughtfully into business operations while fostering a culture of awareness and accountability.

---

### **Key Highlights**

- **AI Risk Education**: Designed to address questions about AI risks, ethical considerations, and mitigation strategies.  
- **Responsible AI Adoption**: Supports responsible AI adoption and risk management across the organization.  
- **Cross-Team Collaboration**: Promotes cross-team learning and collaboration on AI-driven solutions.  

---


## Key Features ✨



- **RAG Pipeline**: A robust Retrieval-Augmented Generation system to answer user queries based on AI risk management frameworks.
- **Fine-Tuned Embeddings**: Custom fine-tuned embeddings using the `snowflake-arctic-embed-m` model for improved performance on domain-specific tasks.
- **Evaluation Framework**: Comprehensive evaluation using the RAGAS framework to measure faithfulness, answer relevancy, context precision, and context recall.
- **Deployment**: The chatbot is deployed on Hugging Face Spaces for easy access and testing.

---
![RAG Application](img/RAG%20Application.png)

---

## Repository Structure 📂

```
Legal-AI-Risk-Management/
├── data/                   # Contains datasets and evaluation results
│   ├── docs_for_rag/       # Documents used for the RAG pipeline (e.g., PDFs, text files)
│   ├── finetuning_data/    # Data used for fine-tuning the embedding model
│   └── rag_questions_and_answers/  # Questions and answers for RAG evaluation and testing
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

## LLM App Stack 🛠️

The chatbot is built using the following **LLM app stack**:

![LLM App Stack Diagram](img/LLM%20App.png)

- **Document Loader**: 
  - **PyMuPDF**: For loading and parsing PDF documents.
- **Text Splitting**:
  - **RecursiveCharacterTextSplitter**: Default chunking strategy for general-purpose text splitting.
  - **Semantic Chunking**: Alternative strategy for grouping semantically related content.
- **Vector Store**:
  - **Qdrant**: In-memory vector store for efficient retrieval.
- **Embeddings**:
  - **OpenAI `text-embedding-3-small`**: Baseline embeddings for general-purpose tasks.
  - **Fine-Tuned `snowflake-arctic-embed-m`**: Custom embeddings fine-tuned for domain-specific performance.
- **LLM**:
  - **OpenAI `gpt-4o-mini`**: High-performance and cost-effective LLM for generating responses.
- **Web Framework**:
  - **Chainlit**: Easy-to-use framework for building and deploying LLM-based web apps.
- **Hosting**:
  - **Hugging Face Spaces**: For deploying and sharing the chatbot.

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
| BaselineChunkArcticOrig           | 0.735739     | 0.869516         | 0.745694          | 0.728333       |
| BaselineChunkArcticFinetuned      | 0.877788     | 0.970543         | 0.985000          | 0.879167       |
| SemanticChunkArcticOrig           | 0.290917     | 0.297116         | 0.391042          | 0.204167       |
| SemanticChunkArcticFinetuned      | 0.896071     | 0.969438         | 0.933403          | 0.916667       |


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

For questions or feedback, please reach out to me on [LinkedIn](https://www.linkedin.com/in/rithyvira/).

---

Happy exploring! 🚀
