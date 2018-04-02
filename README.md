# Temporal TF-IDF Term-Document Scores
TF-IDF scores for documents produced over time

## Equation

$$S(w, t) = F(w, t) * IF(W)$$

$$F(w, p) =$$ frequency of word $$w$$ in all documents in time span $$t$$

$$IF(w) = log(1 / C(w))$$ where $$C(w)$$ is the count of word $$w$$ in the entire collection of documents

## Reference

Viegas, Fernanda B., Scott Golder, and Judith Donath. "Visualizing Email Content: Portraying Relationships from Conversational Histories." _Proceedings of ACM CHI_, April 2006.
