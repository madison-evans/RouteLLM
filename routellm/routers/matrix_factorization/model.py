import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel
from routellm.routers.similarity_weighted.utils import OPENAI_CLIENT

MODEL_IDS = {
    "RWKV-4-Raven-14B": 0,
    "alpaca-13b": 1,
    "chatglm-6b": 2,
    "chatglm2-6b": 3,
    "chatglm3-6b": 4,
    "claude-1": 5,
    "claude-2.0": 6,
    "claude-2.1": 7,
    "claude-instant-1": 8,
    "codellama-34b-instruct": 9,
    "deepseek-llm-67b-chat": 10,
    "dolly-v2-12b": 11,
    "dolphin-2.2.1-mistral-7b": 12,
    "falcon-180b-chat": 13,
    "fastchat-t5-3b": 14,
    "gemini-pro": 15,
    "gemini-pro-dev-api": 16,
    "gpt-3.5-turbo-0125": 17,
    "gpt-3.5-turbo-0314": 18,
    "gpt-3.5-turbo-0613": 19,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-0125-preview": 21,
    "gpt-4-0314": 22,
    "gpt-4-0613": 23,
    "gpt-4-1106-preview": 24,
    "gpt4all-13b-snoozy": 25,
    "guanaco-33b": 26,
    "koala-13b": 27,
    "llama-13b": 28,
    "llama-2-13b-chat": 29,
    "llama-2-70b-chat": 30,
    "llama-2-7b-chat": 31,
    "llama2-70b-steerlm-chat": 32,
    "mistral-7b-instruct": 33,
    "mistral-7b-instruct-v0.2": 34,
    "mistral-medium": 35,
    "mixtral-8x7b-instruct-v0.1": 36,
    "mpt-30b-chat": 37,
    "mpt-7b-chat": 38,
    "nous-hermes-2-mixtral-8x7b-dpo": 39,
    "oasst-pythia-12b": 40,
    "openchat-3.5": 41,
    "openchat-3.5-0106": 42,
    "openhermes-2.5-mistral-7b": 43,
    "palm-2": 44,
    "pplx-70b-online": 45,
    "pplx-7b-online": 46,
    "qwen-14b-chat": 47,
    "qwen1.5-4b-chat": 48,
    "qwen1.5-72b-chat": 49,
    "qwen1.5-7b-chat": 50,
    "solar-10.7b-instruct-v1.0": 51,
    "stablelm-tuned-alpha-7b": 52,
    "starling-lm-7b-alpha": 53,
    "stripedhyena-nous-7b": 54,
    "tulu-2-dpo-70b": 55,
    "vicuna-13b": 56,
    "vicuna-33b": 57,
    "vicuna-7b": 58,
    "wizardlm-13b": 59,
    "wizardlm-70b": 60,
    "yi-34b-chat": 61,
    "zephyr-7b-alpha": 62,
    "zephyr-7b-beta": 63,
}


class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim,
        num_classes,
        use_proj,
        use_openai_embeddings=True,  # Parameter to choose embedding source
        embedding_model_name=None  # Name of the embedding model
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.use_openai_embeddings = use_openai_embeddings

        # Set the embedding model name
        if self.use_openai_embeddings:
            # Default OpenAI embedding model
            self.embedding_model_name = embedding_model_name or "text-embedding-ada-002"
        else:
            # Default Hugging Face embedding model
            self.embedding_model_name = embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

        if not self.use_openai_embeddings:
            # Initialize Hugging Face tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.embedding_model.eval() 
            self.embedding_model.to(self.get_device())
        else:
            self.embedding_model = None  # Not used for OpenAI embeddings

    def get_device(self):
        return self.P.weight.device

    def get_prompt_embedding(self, prompt):
        if self.use_openai_embeddings:
            # Use OpenAI embeddings
            response = OPENAI_CLIENT.embeddings.create(
                input=[prompt],
                model=self.embedding_model_name
            )
            prompt_embed = response.data[0].embedding
            prompt_embed = torch.tensor(prompt_embed, device=self.get_device())
        else:
            # Use Hugging Face embeddings
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.get_device()) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Mean pooling over the token embeddings
                last_hidden_state = outputs.last_hidden_state
                prompt_embed = last_hidden_state.mean(dim=1).squeeze()
        return prompt_embed

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        prompt_embed = self.get_prompt_embedding(prompt)

        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))