import streamlit as st
import torch
import pickle

# --------------------------
# Load your trained model
# --------------------------
# Adjust these lines depending on how you saved your model
MODEL_PATH = "nmt_model.pt"
SRC_VOCAB_PATH = "src_vocab.pkl"
TRG_VOCAB_PATH = "trg_vocab.pkl"

# Load vocab
with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TRG_VOCAB_PATH, "rb") as f:
    trg_vocab = pickle.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Helper: Convert text to tensor
def encode_sentence(sentence, vocab):
    tokens = sentence.lower().split()
    indices = [vocab.get(token, vocab.get("<unk>")) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(device)

# Helper: Translate
def translate_sentence(sentence):
    src_tensor = encode_sentence(sentence, src_vocab)

    with torch.no_grad():
        output = model.translate(src_tensor)  # Ensure your model has a .translate method
    translated_tokens = [trg_vocab.get(idx, "") for idx in output]
    return " ".join(translated_tokens)


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="English → Arabic NMT", page_icon="🌍")

st.title("🌍 English → Arabic Neural Machine Translation")
st.write("Enter an English sentence below and get its Arabic translation.")

# Input box
english_text = st.text_area("✍️ Enter English text:", "")

if st.button("Translate"):
    if english_text.strip():
        arabic_translation = translate_sentence(english_text)
        st.success(f"**Arabic Translation:** {arabic_translation}")
    else:
        st.warning("Please enter some text to translate.")
