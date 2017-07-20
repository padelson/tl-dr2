# 2l-dr
tl-dr (abstractive text summarization), take 2

## Example results
Actual headline: dpp chairman touts 'three direct links' with mainland china
Generated headline: dpp chairman says china will improve ties with taiwan <\s>

Actual headline: two palestinians killed in gaza strip clash with israeli army (updates with toll downgraded by israel, adds jericho arrest)
Generated headline: two palestinians killed in gaza strip <\s>

### Why is it called 2l-dr?
We've tried tackling the problem of text summarization before
(our intial attempt was calle tl-dr), and this is our second attempt.  

### Models
We implemented a GRU encoder-decoder model, and a QRNNenc + RNNdec model.
