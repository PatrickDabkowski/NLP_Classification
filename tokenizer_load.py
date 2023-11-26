from transformers import AutoTokenizer, GPT2Tokenizer

def load(model_name = "bigcode/starcoder", token_path="token.txt", info=False):
    # !!! the starcoder requires the token from Hugging Face
    # I do not provide the token
    # set token_path for None for token free tokenizer

    if token_path:

        with open(token_path, 'r') as file:
            token = [line.strip() for line in file.readlines()]

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        print("tokenizer loaded")
        if info:
            print(tokenizer)
        return tokenizer

    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("tokenizer loaded")
        if info:
            print(tokenizer)
        return tokenizer

if __name__ != "__main__":
    print(f"file: {__name__}")

if __name__ == "__main__":
    # gpt2 tokenizer (without Hugging Face token)
    load("gpt2", None, True)