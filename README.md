# Aicraft-recognition-chatbot

This project uses computer vision to identify aircraft and an LLM chatbot using RAG to answer questions about the identified aircraft

**Dataset**
Source: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz

Total number of images: 3334
Total number of unique classes: 100
Classes are balanced: 
  - max number of photos per class: 34
  - min number of photos per class: 33
  - mean number of photos per class: 33.3
Dataset is clean - no broken images

**Objective**

Predict aircraft model, and return information using an LLM chatbot.

**Training**

Models are trained using **PyTorch** with the following configuration:

- **Loss function:** Cross Entropy Loss
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** StepLR — learning rate halved every 5 epochs (gamma=0.5)

Each epoch consists of two phases:

**Training phase** — the model processes batches of images, computes loss, and updates weights via backpropagation.

**Validation phase** — the model is evaluated on unseen data with gradients disabled. No weight updates occur here.

At the end of each epoch, train/val loss and accuracy are logged. The best model (highest validation accuracy) is automatically saved to Google Drive.

Number of epochs: 5

**Best metrics**

EfficentNetB0: Epoch 05/5 | Train Loss: 2.4077 | Train Acc: 0.5688 | Val Loss: 2.6674   | Val Acc: 0.3580

Resnet50: Epoch 05/5 | Train Loss: 0.2800 | Train Acc: 0.9799 | Val Loss: 1.7922   | Val Acc: 0.5480

**Evaluation**

Models are evaluated on the held-out **test set** (15% of total data) using two approaches:

**Single model evaluation** — each model processes test images independently, taking the class with the highest logit as the final prediction.

**Ensemble evaluation** — predictions from EfficientNetB0 and ResNet50 are combined by averaging their softmax probabilities. The class with the highest average probability is the final prediction. This approach consistently outperforms either individual model.

**Evaluation results**

EfficientNetB0 accuracy: 0.3553

ResNet50 accuracy: 0.5549

Ensemble accuracy: 0.5529

**Additional metrics**
- Confidence distribution
- Model agreement analysis

**LLM chatbot with RAG**

After an aircraft is identified, users can ask questions about it through a conversational chatbot powered by **Mistral AI** and **Retrieval Augmented Generation (RAG)**.

**1. Context loading**
When an aircraft is identified, its Wikipedia article is automatically fetched, chunked, and embedded into a vector store. This becomes the knowledge base for the chatbot session.

**2. Retrieval (RAG)**
When the user asks a question, the most relevant chunks from the Wikipedia article are retrieved using semantic similarity search. Only the most relevant chunks are passed to the LLM, keeping the context focused and reducing hallucinations.

**3. Conversation**
The full chat history is passed to Mistral on every message, allowing the model to maintain context across multiple turns. A system prompt instructs the model to prioritise the retrieved Wikipedia context but fall back to general knowledge if the answer is not found.

**4. Response**
Mistral generates a response grounded in the retrieved context and returns it to the user.

**Architecture**

User question -> Semantic search over Wikipedia chunks -> Top 3 relevant chunks retrieved -> (System prompt + chat history + chunks + question) -> Mistral API -> Answer returned to the user

**Design decisions**

| Wikipedia as a knowledge source | Free, comprehensive, and covers all major aircraft

| RAG over full article | Keeps context window small and answers focused 

| Chat history included | Allows follow-up questions and multi-turn conversation

**API security**

The Mistral API key is stored in Colab Secrets and never hardcoded in the notebook.

**Conclusion**

This project demonstrates an end-to-end deep learning pipeline for fine-grained aircraft recognition:
- Proper data exploration, cleaning, and preprocessing of the FGVC-Aircraft dataset
- Fine-tuning of pretrained models (EfficientNetB0 and ResNet50) for 100-class aircraft classification
- Ensemble learning to improve accuracy beyond individual model performance
- RAG-powered conversational chatbot grounded in Wikipedia knowledge
- Interactive Gradio interface for real-world inference

**Technologies used**

| Deep Learning | PyTorch, torchvision

| Models | EfficientNetB0, ResNet50

| Data Management | FiftyOne, Pickle

| Image Processing | Torchvision transforms 

| RAG & Embeddings | sentence-transformers, Wikipedia API

| LLM | Mistral AI 

| Interface | Gradio 

| Experiment Tracking | Matplotlib, Seaborn, scikit-learn

| Environment | Google Colab, Google Drive 

| Language | Python 
