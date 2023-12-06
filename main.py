from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

ckpt = 'Narrativa/mbart-large-50-finetuned-opus-en-pt-translation'

tokenizer = MBart50TokenizerFast.from_pretrained(ckpt)
model = MBartForConditionalGeneration.from_pretrained(ckpt).to("cpu")

tokenizer.src_lang = 'en_XX'
tokenizer.tgt_lang = 'pt_XX'
def translate(text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids.to('cpu')
    attention_mask = inputs.attention_mask.to('cpu')
    output = model.generate(input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id['pt_XX'])
    return tokenizer.decode(output[0], skip_special_tokens=True)
    
    
s=translate('My name is joão Pedro')
print(s)
